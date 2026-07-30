"""开发用静态服务器。

    python -m scripts.serve            # http://localhost:8123/web/index.html
    python -m scripts.serve --port 9000

和 ``python -m http.server`` 的唯一区别：**每个响应都带 ``Cache-Control: no-store``**。

为什么需要：``http.server`` 会发 ``Last-Modified``，浏览器据此缓存 css/js。
改完样式刷新页面看不到变化，而且症状极具迷惑性 —— DevTools 里
``document.styleSheets`` 能查到新规则（那是 JS 后来抓的那份），
但实际渲染用的是缓存的旧版，于是"规则明明在、就是不生效"。
2026-07-30 在这上面浪费了半小时，从此开发一律走这个脚本。

服务根目录是仓库根，所以 ``/web/`` 和 ``/results/`` 都能访问
（前端的项目状态面板要读 ``results/*.json``）。
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
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


def main() -> int:
    p = argparse.ArgumentParser(description="开发用静态服务器（禁用缓存）")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--bind", default="127.0.0.1")
    args = p.parse_args()

    handler = functools.partial(NoCacheHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.bind, args.port), handler) as httpd:
        url = f"http://localhost:{args.port}/web/index.html"
        print(f"服务根目录：{ROOT}")
        print(f"混音台：    {url}")
        print("（已禁用缓存，改完 css/js 直接刷新即可）  Ctrl-C 停止")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
