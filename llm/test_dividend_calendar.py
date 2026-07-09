# -*- coding: utf-8 -*-
"""Unit tests for the forward dividend-calendar body parser (build_dividend_calendar_upcoming).

Snippets below are verbatim phrasings observed in real e-disclosure substantial-fact bodies for the
16-name universe (2026 FY2025 cycle); they pin the record-date / value / declined extraction so the
fragile regexes can't silently regress. Run: python -m pytest llm/test_dividend_calendar.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import build_dividend_calendar_upcoming as b  # noqa: E402


# ---- record date: phrase "право на получение дивидендов" gates it; date may be before or after ----
@pytest.mark.parametrize("text,expected", [
    # LKOH AGM — date AFTER the phrase, Russian month name
    ("Установить дату, на которую определяются лица, имеющие право на получение дивидендов по "
     "результатам 2025 года, - 4 мая 2026 г.", "2026-05-04"),
    # ROSN board — date after, dash separator
    ("Определить дату, на которую определяются лица, имеющие право на получение дивидендов, "
     "– 09 июля 2026 года.", "2026-07-09"),
    # SBER board — date BEFORE the phrase
    ("утвердить 20 июля 2026 года датой, на которую определяются лица, имеющие право на получение "
     "дивидендов за 2025 год.", "2026-07-20"),
    # PLZL board — date before, "установить DATE датой"
    ("Предложить ... установить 13 июля 2026 года датой, на которую определяются лица, имеющие "
     "право на получение дивидендов по результатам 1 квартала 2026 года.", "2026-07-13"),
    # VTBR board — date well after the phrase ("является DATE")
    ("датой, на которую определяются лица, имеющие право на получение дивидендов, указанных в "
     "пункте 1 настоящего решения, является 20 июля 2026 года.", "2026-07-20"),
    # SNGS — agenda mention (no date) must be ignored; resolution date wins
    ("по дате, на которую определяются лица, имеющие право на получение дивидендов» решение принято. "
     "Установить 16 июля 2026 года в качестве даты, на которую определяются лица, имеющие право на "
     "получение дивидендов.", "2026-07-16"),
])
def test_record_date(text, expected):
    assert b.extract_record_date(text) == expected


def test_record_date_voting_date_excluded():
    # AGM voting record date uses "право голоса", not "право на получение дивидендов" -> not a match
    voting = ("Дата, на которую определяются (фиксируются) лица, имеющие право голоса при принятии "
              "решений общим собранием акционеров: 01 июня 2026 года.")
    assert b.extract_record_date(voting) is None


# ---- per-(ordinary)-share value across the real phrasing variants ----
@pytest.mark.parametrize("text,expected", [
    ("в размере 278 рублей на одну обыкновенную акцию", 278.00),          # LKOH plain
    ("в размере 2 руб. 27 коп. (два рубля двадцать семь копеек) на одну размещенную акцию", 2.27),  # ROSN
    ("в размере 47,23 (сорок семь рублей 23 копейки) рублей на одну обыкновенную акцию", 47.23),    # NVTK
    ("в размере 9,71 рубля на одну размещенную обыкновенную акцию", 9.71),  # VTBR
    ("в размере 56 (пятьдесят шесть) рублей 80 (восемьдесят) копеек на одну обыкновенную акцию", 56.80),  # PLZL
    ("по обыкновенным акциям ПАО Сбербанк – 37,64 руб. на одну акцию", 37.64),  # SBER ("на одну акцию")
    ("в размере 29 (двадцать девять) рублей 05 (пять) копеек на одну обыкновенную акцию", 29.05),  # PLZL Q1
])
def test_value(text, expected):
    val, _ = b.extract_value(text)
    assert val == pytest.approx(expected)


def test_value_prefers_payout_over_annual_total():
    # TATN states the FY total then the installment actually paid at this record ("Произвести выплату")
    tatn = ("Установить общий размер дивиденда за 2025 год: на одну обыкновенную акцию в размере "
            "34 рубля 09 копеек (в том числе дивиденд, объявленный по результатам 6 и 9 месяцев "
            "отчетного года, в размере 22 рубля 48 копеек). Произвести выплату дивидендов: на одну "
            "обыкновенную акцию в размере 11 рублей 61 копейка.")
    val, incl = b.extract_value(tatn)
    assert val == pytest.approx(11.61)
    assert incl is True


# ---- preferred-share value: the pref line reads the SAME body but extracts its own amount ----
@pytest.mark.parametrize("text,expected", [
    ("по обыкновенным акциям ПАО Сбербанк – 37,64 руб. на одну акцию, по привилегированным акциям "
     "ПАО Сбербанк – 37,64 руб. на одну акцию.", 37.64),                                    # SBERP
    ("по привилегированной акции ПАО «Сургутнефтегаз» – 0,85 рубля, по обыкновенной акции "
     "ПАО «Сургутнефтегаз» – 0,85 рубля", 0.85),                                            # SNGSP
])
def test_preferred_value(text, expected):
    val, _ = b.extract_value(text, "preferred")
    assert val == pytest.approx(expected)


def test_preferred_does_not_grab_ordinary_when_they_differ():
    # synthetic year where pref != ord: each anchor must read its OWN amount
    txt = ("по привилегированной акции – 8,50 рубля, по обыкновенной акции – 0,90 рубля")
    assert b.extract_value(txt, "preferred")[0] == pytest.approx(8.50)
    assert b.extract_value(txt, "ordinary")[0] == pytest.approx(0.90)


# ---- declined dividends: detected, and (crucially) carry no dividend record date ----
@pytest.mark.parametrize("text", [
    "дивиденды по акциям ПАО «Газпром» не объявлять и не выплачивать.",                 # GAZP
    "Прибыль по результатам 2025 года не распределять, дивиденды не выплачивать.",      # CHMF/NLMK
    "Дивиденды по результатам 2025 года не выплачивать (не объявлять).",                # ALRS
    "Дивиденды по обыкновенным именным акциям ПАО «Магнит» по результатам 2025 "
    "отчетного года не выплачивать.",                                                  # MGNT
])
def test_decline_detected(text):
    assert b.RE_DECLINE.search(text) is not None
    assert b.extract_record_date(text) is None
