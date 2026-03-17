#!/usr/bin/env python3
# usage: python3 fetch_logs.py
# output: reports/metrica-sessions-<date1>-<date2>.tsv
# install: pip install tapi-yandex-metrika

from tapi_yandex_metrika import YandexMetrikaLogsapi
from datetime import date, timedelta
import csv, sys

TOKEN   = "token"
COUNTER = "counter"

DATE2 = date.today() - timedelta(days=1)
DATE1 = DATE2 - timedelta(days=30)

PARAMS = {
    "fields": ",".join([
        "ym:s:visitID",
        "ym:s:date",
        "ym:s:clientID",
        "ym:s:isNewUser",         # revisit rate
        "ym:s:visitDuration",     # time on site
        "ym:s:goalsID",           # which goals fired per session
        "ym:s:parsedParamsKey1",  # game param keys   e.g. ab_group, level_status
        "ym:s:parsedParamsKey2",  # game param values e.g. A, completed
    ]),
    "source": "visits",
    "date1": DATE1,
    "date2": DATE2,
}

client = YandexMetrikaLogsapi(
    access_token=TOKEN,
    default_url_params={"counterId": COUNTER},
    wait_report=True,
)

print(f"Evaluating · counter {COUNTER} · {DATE1} → {DATE2}")
evaluation = client.evaluate().get(params=PARAMS)
if not evaluation["log_request_evaluation"]["possible"]:
    print("Not enough data yet — try again tomorrow.")
    sys.exit(0)

print("Creating report…")
report = client.create().post(params=PARAMS)
request_id = report["log_request"]["request_id"]
print(f"Request ID: {request_id}")

out = f"reports/metrica-sessions-{DATE1}-{DATE2}.tsv"
rows = list(client.download(requestId=request_id, partNumber=0).get().iter_dicts())

if not rows:
    print("No rows returned.")
    sys.exit(0)

with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} sessions → {out}")

client.clean(requestId=request_id).post()
print("Log request cleaned.")
