#!/usr/bin/env python3
"""Trading 212 paper-trading terminal, permanently locked to the Demo API."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEMO_BASE_URL = "https://demo.trading212.com/api/v0"
ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / ".cache" / "instruments.json"


class Trading212Error(RuntimeError):
    """A readable Trading 212 API error."""


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load a minimal KEY=VALUE file without overriding existing variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


class Trading212DemoClient:
    """Small standard-library client for the Trading 212 Demo environment."""

    def __init__(self, api_key: str, api_secret: str, timeout: float = 15.0):
        if not api_key or not api_secret:
            raise ValueError("缺少 T212_API_KEY 或 T212_API_SECRET")
        token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
            "User-Agent": "t212-demo-terminal/1.0",
        }
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        url = DEMO_BASE_URL + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None
        headers = dict(self._headers)
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.dumps(json.loads(raw), ensure_ascii=False)
            except json.JSONDecodeError:
                detail = raw or exc.reason
            hint = ""
            if exc.code == 401:
                hint = "；请确认使用的是 Demo 账户生成的 Key 与 Secret"
            elif exc.code == 403:
                hint = "；请确认 API 权限包含所需的账户/持仓/订单权限"
            elif exc.code == 429:
                hint = "；请求过快，请稍后重试"
            raise Trading212Error(f"Trading 212 返回 HTTP {exc.code}: {detail}{hint}") from exc
        except urllib.error.URLError as exc:
            raise Trading212Error(f"无法连接 Trading 212 Demo API: {exc.reason}") from exc

    def account_summary(self) -> dict[str, Any]:
        return self.request("GET", "/equity/account/summary")

    def positions(self, ticker: str | None = None) -> list[dict[str, Any]]:
        query = {"ticker": ticker} if ticker else None
        return self.request("GET", "/equity/positions", query=query)

    def pending_orders(self) -> list[dict[str, Any]]:
        return self.request("GET", "/equity/orders")

    def instruments(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if not refresh and CACHE_FILE.exists():
            age = time.time() - CACHE_FILE.stat().st_mtime
            if age < 600:
                return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        result = self.request("GET", "/equity/metadata/instruments")
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    def market_order(self, ticker: str, signed_quantity: float, extended_hours: bool) -> dict[str, Any]:
        return self.request(
            "POST",
            "/equity/orders/market",
            body={
                "ticker": ticker,
                "quantity": signed_quantity,
                "extendedHours": extended_hours,
            },
        )

    def limit_order(
        self,
        ticker: str,
        signed_quantity: float,
        limit_price: float,
        time_validity: str,
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            "/equity/orders/limit",
            body={
                "ticker": ticker,
                "quantity": signed_quantity,
                "limitPrice": limit_price,
                "timeValidity": time_validity,
            },
        )

    def cancel_order(self, order_id: int) -> Any:
        return self.request("DELETE", f"/equity/orders/{order_id}")


def positive_number(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是数字") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数字")
    return number


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def nested(value: dict[str, Any], *keys: str, default: Any = "-") -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def render_positions(positions: list[dict[str, Any]]) -> None:
    if not positions:
        print("当前没有持仓。")
        return
    print(f"{'Ticker':<20} {'数量':>12} {'现价':>12} {'市值':>14} {'浮盈亏':>14}")
    print("-" * 76)
    for item in positions:
        ticker = nested(item, "instrument", "ticker", default=item.get("ticker", "-"))
        print(
            f"{str(ticker):<20} "
            f"{item.get('quantity', '-'):>12} "
            f"{item.get('currentPrice', '-'):>12} "
            f"{nested(item, 'walletImpact', 'currentValue'):>14} "
            f"{nested(item, 'walletImpact', 'unrealizedProfitLoss'):>14}"
        )


def render_orders(orders: list[dict[str, Any]]) -> None:
    if not orders:
        print("当前没有待成交订单。")
        return
    print(f"{'ID':<14} {'Ticker':<20} {'方向':<6} {'类型':<12} {'数量':>12} {'状态':<18}")
    print("-" * 88)
    for item in orders:
        ticker = nested(item, "instrument", "ticker", default=item.get("ticker", "-"))
        print(
            f"{str(item.get('id', '-')):<14} {str(ticker):<20} "
            f"{str(item.get('side', '-')):<6} {str(item.get('type', '-')):<12} "
            f"{str(item.get('quantity', '-')):>12} {str(item.get('status', '-')):<18}"
        )


def require_demo_confirmation(args: argparse.Namespace, payload: dict[str, Any]) -> bool:
    print("将发送到 Trading 212 模拟账户：")
    print_json(payload)
    if not args.confirm_demo:
        print("未执行。确认无误后在命令末尾添加 --confirm-demo。")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading 212 Demo 模拟交易终端（无法连接真实账户）")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="验证 Demo 凭证并显示账户摘要")

    positions = sub.add_parser("positions", help="显示持仓和当前价格")
    positions.add_argument("--ticker", help="只显示指定 Trading 212 ticker")
    positions.add_argument("--watch", action="store_true", help="持续刷新")
    positions.add_argument("--interval", type=positive_number, default=1.2, help="刷新秒数，最小 1.1")

    instruments = sub.add_parser("search", help="搜索可交易标的")
    instruments.add_argument("query", help="名称、简称、ticker 或 ISIN")
    instruments.add_argument("--limit", type=int, default=20)
    instruments.add_argument("--refresh", action="store_true", help="忽略 10 分钟本地缓存")

    sub.add_parser("orders", help="显示待成交订单")

    market = sub.add_parser("market", help="提交模拟市价单")
    market.add_argument("side", choices=("buy", "sell"))
    market.add_argument("ticker")
    market.add_argument("quantity", type=positive_number)
    market.add_argument("--extended-hours", action="store_true")
    market.add_argument("--confirm-demo", action="store_true")

    limit_order = sub.add_parser("limit", help="提交模拟限价单")
    limit_order.add_argument("side", choices=("buy", "sell"))
    limit_order.add_argument("ticker")
    limit_order.add_argument("quantity", type=positive_number)
    limit_order.add_argument("price", type=positive_number)
    limit_order.add_argument("--validity", choices=("DAY", "GOOD_TILL_CANCEL"), default="DAY")
    limit_order.add_argument("--confirm-demo", action="store_true")

    cancel = sub.add_parser("cancel", help="撤销模拟待成交订单")
    cancel.add_argument("order_id", type=int)
    cancel.add_argument("--confirm-demo", action="store_true")
    return parser


def make_client() -> Trading212DemoClient:
    load_dotenv()
    key = os.environ.get("T212_API_KEY", "")
    secret = os.environ.get("T212_API_SECRET", "")
    if not key or not secret:
        raise Trading212Error(
            "尚未配置 Demo 凭证。请复制 .env.example 为 .env，"
            "并在本机填入 Demo API Key 与 Secret。不要把密钥发到聊天中。"
        )
    return Trading212DemoClient(key, secret)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = make_client()
        if args.command == "check":
            print(f"环境：{DEMO_BASE_URL}（模拟盘）")
            print_json(client.account_summary())
        elif args.command == "positions":
            interval = max(args.interval, 1.1)
            while True:
                if args.watch:
                    print("\033[2J\033[H", end="")
                    print(f"Trading 212 Demo 持仓 · {datetime.now().astimezone().isoformat(timespec='seconds')}")
                render_positions(client.positions(args.ticker))
                if not args.watch:
                    break
                time.sleep(interval)
        elif args.command == "search":
            query = args.query.casefold()
            matches = []
            for item in client.instruments(refresh=args.refresh):
                haystack = " ".join(str(item.get(k, "")) for k in ("ticker", "name", "shortName", "isin"))
                if query in haystack.casefold():
                    matches.append(item)
            print_json(matches[: max(args.limit, 0)])
        elif args.command == "orders":
            render_orders(client.pending_orders())
        elif args.command == "market":
            signed = args.quantity if args.side == "buy" else -args.quantity
            payload = {
                "environment": "DEMO",
                "type": "MARKET",
                "ticker": args.ticker,
                "quantity": signed,
                "extendedHours": args.extended_hours,
            }
            if require_demo_confirmation(args, payload):
                print_json(client.market_order(args.ticker, signed, args.extended_hours))
        elif args.command == "limit":
            signed = args.quantity if args.side == "buy" else -args.quantity
            payload = {
                "environment": "DEMO",
                "type": "LIMIT",
                "ticker": args.ticker,
                "quantity": signed,
                "limitPrice": args.price,
                "timeValidity": args.validity,
            }
            if require_demo_confirmation(args, payload):
                print_json(client.limit_order(args.ticker, signed, args.price, args.validity))
        elif args.command == "cancel":
            payload = {"environment": "DEMO", "action": "CANCEL", "orderId": args.order_id}
            if require_demo_confirmation(args, payload):
                result = client.cancel_order(args.order_id)
                print_json(result if result is not None else {"cancelled": True, "id": args.order_id})
        return 0
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130
    except (Trading212Error, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
