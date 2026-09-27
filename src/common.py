from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib

from rapidfuzz import fuzz
from unidecode import unidecode

LEGAL = {
    "inc", "incorporated", "corporation", "corp", "company", "co", "limited", "ltd",
    "llc", "llp", "private", "pvt", "plc", "sarl", "sas", "gmbh",
}
WS = re.compile(r"\s+")
NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)
DIGITS = re.compile(r"\d+")
NAME_SEEDS = (17, 7919, 104729, 130363, 2147483647)
ADDRESS_SEEDS = (31, 8191, 65537, 99991)
TOKEN_SEEDS = (41, 65539)

FEATURE_NAMES = [
    "name_exact", "name_base_exact", "address_exact", "name_ratio", "name_wratio",
    "name_token_set", "name_partial", "name_jaccard", "address_ratio", "address_wratio",
    "address_token_set", "address_partial", "address_jaccard", "digit_jaccard",
    "digit_signature_exact", "name_length_ratio", "address_length_ratio", "address_missing",
    "source3", "block_evidence_count",
]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().replace("&", " and ")
    return WS.sub(" ", NON_ALNUM.sub(" ", value)).strip()


def base_name(value: str) -> str:
    return " ".join(token for token in normalize(value).split() if token not in LEGAL)


def ascii_compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(base_name(value)).casefold())


def address_compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(normalize(value)).casefold())


def soundex(word: str) -> str:
    word = re.sub(r"[^a-z]", "", unidecode(word).casefold())
    if not word:
        return ""
    mapping = str.maketrans("bfpvcgjkqsxzdtlmnr", "111122222222334556")
    collapsed, previous = [], ""
    for char in word[1:].translate(mapping):
        if char != previous:
            collapsed.append(char)
        previous = char
    digits = "".join(char for char in collapsed if char.isdigit())
    return (word[0] + digits + "000")[:4]


def phonetic_signature(value: str) -> str:
    return " ".join(sorted(filter(None, (soundex(token) for token in base_name(value).split()))))


def token_minhash(value: str, seeds=TOKEN_SEEDS) -> list[int | None]:
    tokens = sorted(set(re.findall(r"[a-z0-9]+", unidecode(normalize(value)).casefold())))
    if not tokens:
        return [None] * len(seeds)
    payload = [token.encode("utf-8") for token in tokens]
    return [min(zlib.crc32(token, seed) & 0xFFFFFFFF for token in payload) for seed in seeds]


def digit_signature(value: str) -> str:
    return " ".join(sorted(set(x.lstrip("0") or "0" for x in DIGITS.findall(value or "") if len(x) >= 3)))


def minhash_grams(text: str, n: int, seeds: tuple[int, ...]) -> list[int | None]:
    if not text:
        return [None] * len(seeds)
    n = min(n, len(text))
    grams = [text[i : i + n].encode("utf-8") for i in range(len(text) - n + 1)]
    return [min(zlib.crc32(gram, seed) & 0xFFFFFFFF for gram in grams) for seed in seeds]


def hash63(value: str) -> int | None:
    if not value:
        return None
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big") & ((1 << 63) - 1)


def record_keys(name: str, address: str) -> dict[str, int | None]:
    name_norm = normalize(name)
    base = base_name(name)
    signature = " ".join(sorted(base.split()))
    ascii_name = ascii_compact(name)
    address_norm = normalize(address)
    digits = digit_signature(address)
    values = {
        "k_name": hash63(name_norm),
        "k_base": hash63(base),
        "k_signature": hash63(signature),
        "k_ascii": hash63(ascii_name),
        "k_address": hash63(address_norm),
        "k_digits": hash63(digits),
        "k_prefix": hash63(ascii_name[:10] if len(ascii_name) >= 8 else ""),
        "k_phonetic": hash63(phonetic_signature(name)),
    }
    for i, value in enumerate(minhash_grams(ascii_name, 4, NAME_SEEDS)):
        values[f"name_mh{i}"] = value
    for i, value in enumerate(minhash_grams(address_compact(address), 6, ADDRESS_SEEDS)):
        values[f"address_mh{i}"] = value
    for i, value in enumerate(token_minhash(name)):
        values[f"name_token_mh{i}"] = value
    for i, value in enumerate(token_minhash(address)):
        values[f"address_token_mh{i}"] = value
    return values


def jaccard(a: str, b: str) -> float:
    left, right = set(a.split()), set(b.split())
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left and right else 0.0


def digit_jaccard(a: str, b: str) -> float:
    left, right = set(DIGITS.findall(a)), set(DIGITS.findall(b))
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right) if left and right else 0.0


def length_ratio(a: str, b: str) -> float:
    return min(len(a), len(b)) / max(1, max(len(a), len(b)))


def text_views(record: dict) -> tuple[str, str, str, str]:
    return (
        normalize(record["business_name"]), base_name(record["business_name"]),
        normalize(record["business_address"]), digit_signature(record["business_address"]),
    )


def pair_features(left: dict, right: dict, evidence_count: int, left_views=None) -> list[float]:
    ln, lb, la, ld = left_views if left_views is not None else text_views(left)
    rn, rb, ra, rd = text_views(right)
    return [
        float(bool(ln) and ln == rn), float(bool(lb) and lb == rb), float(bool(la) and la == ra),
        fuzz.ratio(ln, rn) / 100.0, fuzz.WRatio(ln, rn) / 100.0,
        fuzz.token_set_ratio(ln, rn) / 100.0, fuzz.partial_ratio(ln, rn) / 100.0,
        jaccard(ln, rn), fuzz.ratio(la, ra) / 100.0 if la and ra else 0.0,
        fuzz.WRatio(la, ra) / 100.0 if la and ra else 0.0,
        fuzz.token_set_ratio(la, ra) / 100.0 if la and ra else 0.0,
        fuzz.partial_ratio(la, ra) / 100.0 if la and ra else 0.0,
        jaccard(la, ra), digit_jaccard(la, ra), float(bool(ld) and ld == rd),
        length_ratio(ln, rn), length_ratio(la, ra), float(not la or not ra),
        float(right["entity_id"].startswith("S3-")), float(evidence_count),
    ]


def cheap_score(left: dict, right: dict, evidence_count: int, left_views=None) -> float:
    ln, lb, la, _ = left_views if left_views is not None else text_views(left)
    rn, rb, ra, _ = text_views(right)
    ns = fuzz.WRatio(ln, rn) / 100.0
    ads = fuzz.WRatio(la, ra) / 100.0 if la and ra else 0.0
    nums = digit_jaccard(la, ra)
    exact = 0.05 * (ln == rn) + 0.04 * (lb == rb)
    return max(0.62 * ns + 0.30 * ads + 0.08 * nums, 0.82 * ns + 0.18 * nums, 0.75 * ads + 0.25 * ns) + exact + 0.01 * min(evidence_count, 5)
