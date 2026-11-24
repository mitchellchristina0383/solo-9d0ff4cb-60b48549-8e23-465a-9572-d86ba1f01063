"""Plan parsing and HTTP surface tests."""

from __future__ import annotations

import pytest

from memdry.app import create_app
from memdry.model import ValidationError, parse_plan


@pytest.fixture()
def client():
    app = create_app({"TESTING": True})
    return app.test_client()


def test_parse_rejects_bad_capacity():
    with pytest.raises(ValidationError) as info:
        parse_plan({"capacity": "0", "inputs": "a: 1", "ops": "b = relu(a)", "outputs": "b"})
    assert any("大于 0" in message for message in info.value.messages)


def test_parse_rejects_add_length_mismatch():
    with pytest.raises(ValidationError) as info:
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2\nb: 1",
                "ops": "c = add(a, b)",
                "outputs": "c",
            }
        )
    assert any("等长" in message for message in info.value.messages)


def test_parse_rejects_stepped_slice_and_unknown_operands():
    with pytest.raises(ValidationError) as info:
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2, 3",
                "ops": "v = slice(a, 0, 3, 2)",
                "outputs": "v",
            }
        )
    assert any("步长" in message for message in info.value.messages)

    with pytest.raises(ValidationError):
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2, 3",
                "ops": "v = relu(nope)",
                "outputs": "v",
            }
        )


def test_parse_rejects_empty_slice_and_duplicate_names():
    with pytest.raises(ValidationError) as info:
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2, 3",
                "ops": "v = slice(a, 2, 2)",
                "outputs": "v",
            }
        )
    assert any("为空" in message for message in info.value.messages)

    with pytest.raises(ValidationError) as info:
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2\nb: 1, 2",
                "ops": "b = relu(a)",
                "outputs": "b",
            }
        )
    assert any("重名" in message for message in info.value.messages)


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "端侧推理" in body
    assert "缓冲容量" in body


def test_healthz(client):
    assert client.get("/healthz").get_json() == {"ok": True}


def test_api_requires_json(client):
    response = client.post("/api/run", data="not json", content_type="text/plain")
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_api_returns_validation_errors(client):
    response = client.post(
        "/api/run",
        json={"capacity": "2", "inputs": "a: 1, 2", "ops": "b = relu(nope)", "outputs": "b"},
    )
    assert response.status_code == 400
    assert response.get_json()["errors"]


def test_api_success_payload(client):
    response = client.post(
        "/api/run",
        json={
            "capacity": "12",
            "inputs": "a: 1, 2, 3\nb: 4, 5, 6",
            "ops": "s = add(a, b)\nv = slice(s, 1, 3)",
            "outputs": "v",
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["outputs"][0]["values"] == ["7", "9"]
    assert body["peak"]["elements"] == 9
    assert body["layout"]


def test_api_insufficient_capacity_is_a_200_report(client):
    response = client.post(
        "/api/run",
        json={
            "capacity": "4",
            "inputs": "a: 1, 2\nb: 3, 4",
            "ops": "s = add(a, b)",
            "outputs": "s",
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is False
    assert body["failure"]["needed"] == 2
    assert body["failure"]["step"] == 3


REORDER_PAYLOAD = {
    "capacity": "21",
    "inputs": (
        "A: 1, -2, 3, -4, 5, -6, 7, -8\n"
        "B: 1, 2, 3, 4\nC: 10, -10, 20, -20"
    ),
    "ops": "x = relu(A)\ny = add(B, C)\nu = relu(y)",
    "outputs": "x, u",
}


def test_api_low_peak_reorders_and_executes(client):
    response = client.post("/api/run", json={**REORDER_PAYLOAD, "low_peak": True})
    assert response.status_code == 200
    body = response.get_json()
    assert body["mode"] == "low_peak"
    assert body["executed"] is True
    info = body["low_peak"]
    assert info["selected"] is True
    assert info["max_ops"] == 12
    assert info["reordered"] is True
    assert info["original_peak"] == 24
    assert info["min_elements"] == 20
    selected = [entry["index"] for entry in info["order"]]
    assert sorted(selected) == [0, 1, 2]
    assert selected in ([1, 2, 0], [1, 0, 2])
    assert info["capacity_sufficient"] is True
    # The reordered run fits where the original order would not.
    assert body["ok"] is True
    assert body["peak"]["elements"] == 20
    values = {out["name"]: out["values"] for out in body["outputs"]}
    assert values["x"] == ["1", "0", "3", "0", "5", "0", "7", "0"]
    assert values["u"] == ["11", "0", "23", "0"]


def test_api_low_peak_reordered_values_match_original_order(client):
    with_order = client.post(
        "/api/run", json={**REORDER_PAYLOAD, "capacity": "100", "low_peak": True}
    ).get_json()
    original = client.post(
        "/api/run", json={**REORDER_PAYLOAD, "capacity": "100"}
    ).get_json()
    a = {o["name"]: o["values"] for o in with_order["outputs"]}
    b = {o["name"]: o["values"] for o in original["outputs"]}
    assert a == b


def test_api_low_peak_skips_execution_when_capacity_short(client):
    response = client.post(
        "/api/run",
        json={**REORDER_PAYLOAD, "capacity": "15", "low_peak": True},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["mode"] == "low_peak"
    assert body["executed"] is False
    assert body["ok"] is False
    assert body["steps"] == []
    assert body["outputs"] == []
    info = body["low_peak"]
    assert info["executed"] is False
    assert info["capacity_sufficient"] is False
    assert info["min_elements"] == 20
    assert "15" in info["skip_reason"] and "20" in info["skip_reason"]
    assert sorted(entry["index"] for entry in info["order"]) == [0, 1, 2]


def test_api_low_peak_keeps_original_order_when_already_optimal(client):
    response = client.post(
        "/api/run",
        json={
            "capacity": "20",
            "inputs": "a: 1, 2, 3\nb: 4, 5, 6",
            "ops": "s = add(a, b)\nr = relu(s)",
            "outputs": "r",
            "low_peak": "true",
        },
    )
    assert response.status_code == 200
    info = response.get_json()["low_peak"]
    assert info["reordered"] is False
    assert [entry["index"] for entry in info["order"]] == [0, 1]


def test_api_low_peak_rejects_more_than_twelve_ops(client):
    ops = "\n".join(f"r{i} = relu({'x' if i == 0 else f'r{i-1}'})" for i in range(13))
    response = client.post(
        "/api/run",
        json={
            "capacity": "100",
            "inputs": "x: 1, 2",
            "ops": ops,
            "outputs": "r12",
            "low_peak": True,
        },
    )
    assert response.status_code == 400
    messages = response.get_json()["errors"]
    assert any("12" in message for message in messages)


def test_api_without_low_peak_unchanged_shape(client):
    response = client.post("/api/run", json=REORDER_PAYLOAD)
    body = response.get_json()
    assert "low_peak" not in body
    assert "mode" not in body
    assert body["ok"] is False  # original order overflows at capacity 21
