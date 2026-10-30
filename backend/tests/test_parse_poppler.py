"""PDF와 Poppler 없이, 추출된 글자에서 블록을 나누는 규칙만 확인한다."""

import re

from scripts.parse_poppler import page_blocks

STRIP = [re.compile(p) for p in [r"^조류경보제 운영 매뉴얼$", r"^\d+$"]]

PAGE = """조류경보제 운영 매뉴얼


  1.3.1   기준항목 : 남조류 세포수
       ※ 남조류 세포수는 네 속(屬) 세포수의 합을 말한다.

  2회 연속 측정하여 발령하고,
2회 연속채취 시 1,000 세포/mL미만인 경우 해제한다.

 표 6      단계별 조류경보 발령기준

       경보단계            발령·해제 기준
          관심   1,000 세포/mL 이상

30
"""


def blocks():
    out, section = page_blocks(PAGE, "DOC02", 34, 30, "1.3 발령 기준", STRIP)
    return out, section


def test_header_and_page_number_removed():
    out, _ = blocks()
    text = "\n".join(b["text"] for b in out)
    assert "조류경보제 운영 매뉴얼" not in text
    assert "\n30" not in text


def test_heading_sets_section_and_is_not_a_block():
    out, section = blocks()
    assert out[0]["section"] == "1.3.1 기준항목 : 남조류 세포수"
    assert out[0]["text"].startswith("※ 남조류")
    assert section == "1.3.1 기준항목 : 남조류 세포수"


def test_paragraph_keeps_line_breaks():
    out, _ = blocks()
    assert out[1]["text"] == (
        "2회 연속 측정하여 발령하고,\n2회 연속채취 시 1,000 세포/mL미만인 경우 해제한다."
    )


def test_caption_only_block_joins_table_body():
    out, _ = blocks()
    table = out[2]
    assert table["type"] == "table"
    assert table["table_id"] == "표 6"
    assert table["caption"] == "단계별 조류경보 발령기준"
    assert "1,000 세포/mL 이상" in table["text"]
    assert len(out) == 3


def test_pages_and_ids():
    out, _ = blocks()
    assert [b["block_id"] for b in out] == [
        "DOC02-p034-b01",
        "DOC02-p034-b02",
        "DOC02-p034-b03",
    ]
    assert all(b["printed_page_start"] == 30 for b in out)
    assert all(b["review"] == {"status": "raw"} for b in out)


def test_number_led_sentence_is_not_heading():
    out, section = page_blocks(
        "2020년 수질예보제와 조류경보제가 통합되면서\n조류예측을 실시하고 있다.\n",
        "DOC01",
        174,
        154,
        "2. 조류예측",
        [],
    )
    assert section == "2. 조류예측"
    assert out[0]["text"].startswith("2020년")
