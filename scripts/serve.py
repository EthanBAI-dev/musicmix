"""开发用服务器：静态文件（禁用缓存）+ 分离任务 API。

    python -m scripts.serve            # http://localhost:8123/web/index.html
    python -m scripts.serve --port 9000

**静态部分**和 ``python -m http.server`` 的区别：每个响应都带
``Cache-Control: no-store``。``http.server`` 会发 ``Last-Modified``，浏览器据此缓存
css/js，改完样式刷新看不到变化 —— 而且症状极具迷惑性：DevTools 里
``document.styleSheets`` 能查到新规则（那是 JS 后来抓的那份），
实际渲染用的却是缓存的旧版。2026-07-30 在这上面浪费了半小时，从此开发一律走这个脚本。

**API 部分**（P6 服务化的最小可用版本）：

- ``POST /api/separate``  原始音频字节，文件名放在 ``X-Filename`` 头里 → ``{job_id}``
- ``GET  /api/jobs/<id>`` 查询进度
- ``GET  /api/jobs``      最近的任务列表

用裸字节而不是 multipart：省掉一个已废弃的 ``cgi`` 依赖，前端也只要
``fetch(url, {method:'POST', body: file})`` 一行。

.. warning::
   **这个服务会接收任意文件并在本机上跑解码与模型推理。**
   因此默认只绑 ``127.0.0.1``，绑到其他地址需要显式加 ``--allow-remote``
   并自行承担后果 —— 不要把它暴露到公网。
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socketserver
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    # ---------- 通用 ----------

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # 默认日志会把每个 mp3 分片请求都刷出来，噪音太大；只记非 2xx
        status = str(args[1]) if len(args) > 1 else ""
        if not status.startswith("2"):
            super().log_message(fmt, *args)

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------- API ----------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/jobs":
            from src.service.jobs import recent_jobs
            return self._json({"jobs": recent_jobs()})
        if path.startswith("/api/jobs/"):
            from src.service.jobs import get_job
            job = get_job(path.rsplit("/", 1)[-1])
            return self._json(job.as_dict() if job else {"error": "无此任务"},
                              200 if job else 404)
        return super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/separate":
            return self._json({"error": "未知接口"}, 404)

        from src.service.jobs import MAX_UPLOAD_BYTES, submit

        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json({"error": "Content-Length 非法"}, 400)
        if size <= 0:
            return self._json({"error": "空请求体"}, 400)
        if size > MAX_UPLOAD_BYTES:
            return self._json(
                {"error": f"文件过大（{size/2**20:.0f} MB），上限 {MAX_UPLOAD_BYTES/2**20:.0f} MB"}, 413)

        raw = self.rfile.read(size)
        # 头里是百分号编码的（HTTP 头只能带 ASCII），解回来才能拿到中文名
        filename = unquote(self.headers.get("X-Filename") or "upload.mp3")
        # 只取 basename：headers 是客户端可控的，别让 ../ 之类的东西进到路径里
        filename = Path(filename).name

        model = self.headers.get("X-Model") or "htdemucs"
        if model not in ("htdemucs", "htdemucs_ft", "mdx_extra"):
            return self._json({"error": f"不支持的模型 {model}"}, 400)
        try:
            seconds = float(self.headers.get("X-Seconds") or 0)
        except ValueError:
            seconds = 0.0

        job = submit(raw, filename, model=model, seconds=seconds)
        return self._json(job.as_dict(), 202)


def main() -> int:
    p = argparse.ArgumentParser(description="开发用服务器（静态 + 分离 API）")
    # 端口优先级：命令行 > $PORT > 默认。
    # 认 $PORT 是为了让托管方（如编辑器的预览面板）能分配空闲端口 ——
    # 本机上 8000 已被另一个项目的服务长期占用，写死端口迟早撞车。
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8123)))
    p.add_argument("--bind", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--allow-remote", action="store_true",
                   help="允许绑到非本地地址。这个服务会接收任意文件并跑推理，"
                        "除非你清楚风险，否则别开")
    args = p.parse_args()

    if args.bind not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        print(f"❌ 拒绝绑定 {args.bind}：本服务接收任意上传并在本机跑推理，"
              f"只应监听 127.0.0.1。\n   确实需要请显式加 --allow-remote。", file=sys.stderr)
        return 1

    handler = functools.partial(Handler, directory=str(ROOT))

    # ThreadingHTTPServer 是必须的：分离任务跑在后台线程里，
    # 单线程服务器会让轮询进度的请求排在上传请求后面，进度条永远不动。
    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    with Server((args.bind, args.port), handler) as httpd:
        print(f"服务根目录：{ROOT}")
        print(f"混音台：    http://localhost:{args.port}/web/index.html")
        print("（已禁用缓存；上传分离接口 POST /api/separate）  Ctrl-C 停止")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
