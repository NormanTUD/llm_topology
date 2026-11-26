from __future__ import annotations

from fastapi.testclient import TestClient

from llm_topology.web.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "torch" in body


def test_index_renders_lab() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Topology Lab" in response.text
    assert "What the machine sees" in response.text


def test_catalog_contains_structures_and_tokens() -> None:
    response = client.get("/api/catalog")
    assert response.status_code == 200
    body = response.json()
    assert any(item["value"] == "torus" for item in body["structures"])
    assert "<endoftext>" in body["tokens"]
    assert body["boundary_token"] == "<endoftext>"
    assert body["default_spec"]["structure"] == "circle"
    assert any(block["name"] == "structure_id" for block in body["target_blocks"])
    assert body["model_defaults"]["n_layers"] >= 1
    assert body["training_defaults"]["optimizer"] == "adamw"


def test_generate_small_dataset() -> None:
    payload = {
        "structure": "circle",
        "n_points": 60,
        "ambient_dim": 2,
        "seed": 1,
        "filtration": {
            "method": "alpha",
            "max_dim": 2,
            "field": "Z2",
            "epsilon": 0.5,
        },
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sample"]["n_points"] == 60
    assert len(body["target_groups"]) == 1
    assert "<endoftext>" in body["sequence"]
    assert body["projection"]
    assert len(body["projection"][0]) == 3


def test_generate_batch() -> None:
    payload = {
        "spec": {
            "structure": "circle",
            "n_points": 40,
            "ambient_dim": 2,
            "seed": 1,
            "filtration": {"method": "alpha", "max_dim": 2, "epsilon": 0.5},
        },
        "count": 3,
    }
    response = client.post("/api/generate/batch", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 3
    assert len(body["items"]) == 3
    assert body["mode"] == "single"
    assert {item["sample"]["structure"] for item in body["items"]} == {"circle"}
    assert body["items"][0]["target_groups"]


def test_generate_batch_mix_requires_and_uses_structures() -> None:
    spec = {
        "structure": "circle",
        "n_points": 30,
        "ambient_dim": 3,
        "seed": 2,
        "filtration": {"method": "alpha", "max_dim": 2, "epsilon": 0.5},
    }
    missing = client.post("/api/generate/batch", json={"spec": spec, "count": 2, "mode": "mix"})
    assert missing.status_code == 422
    response = client.post(
        "/api/generate/batch",
        json={
            "spec": spec,
            "count": 2,
            "mode": "mix",
            "structures": ["circle", "sphere"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "mix"
    assert [item["sample"]["structure"] for item in body["items"]] == [
        "circle",
        "sphere",
    ]


def test_lab_page() -> None:
    response = client.get("/lab")
    assert response.status_code == 200
    assert "Model Lab" in response.text
    assert "Training loss" in response.text


def test_ui_state_roundtrip() -> None:
    response = client.post("/api/ui-state", json={"test_marker": "ok"})
    assert response.status_code == 200
    assert response.json()["test_marker"] == "ok"
    fetched = client.get("/api/ui-state")
    assert fetched.status_code == 200
    assert fetched.json()["test_marker"] == "ok"


def test_dataset_zip_export_import(monkeypatch, tmp_path) -> None:
    from llm_topology.web.main import DATASET_STORE

    monkeypatch.setattr(DATASET_STORE, "root", tmp_path / "source")
    payload = {
        "name": "export-set",
        "data_spec": {
            "structures": ["circle"],
            "n_train": 4,
            "n_val": 2,
            "n_test": 2,
            "n_points": 8,
            "max_seq_len": 512,
            "precision": 5,
            "augment": False,
        },
    }
    created = client.post("/api/datasets", json=payload)
    dataset_id = created.json()["id"]
    exported = client.get(f"/api/datasets/{dataset_id}/export")
    assert exported.status_code == 200
    archive = exported.content
    monkeypatch.setattr(DATASET_STORE, "root", tmp_path / "target")
    imported = client.post(
        "/api/datasets/import",
        files={"file": ("dataset.zip", archive, "application/zip")},
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["imported"] == dataset_id


def test_dataset_api_roundtrip(monkeypatch, tmp_path) -> None:
    from llm_topology.web.main import DATASET_STORE

    monkeypatch.setattr(DATASET_STORE, "root", tmp_path / "datasets")
    payload = {
        "name": "tiny-api-set",
        "data_spec": {
            "structures": ["circle"],
            "n_train": 4,
            "n_val": 2,
            "n_points": 8,
            "max_seq_len": 512,
            "precision": 2,
            "augment": False,
        },
    }
    response = client.post("/api/datasets", json=payload)
    assert response.status_code == 200, response.text
    meta = response.json()
    assert meta["train_count"] == 4
    assert meta["test_count"] >= 2
    assert any(item["id"] == meta["id"] for item in client.get("/api/datasets").json())
    preview = client.get(f"/api/datasets/{meta['id']}/preview?split=test&limit=2")
    assert preview.status_code == 200
    assert preview.json()["split"] == "test"
    assert len(preview.json()["items"]) <= 2


def test_config_export_import_endpoints() -> None:
    spec = {
        "data": {
            "structures": ["circle"],
            "n_train": 4,
            "n_val": 2,
            "n_test": 2,
            "max_seq_len": 512,
            "precision": 5,
        },
        "model": {
            "max_seq_len": 512,
            "n_layers": 1,
            "n_heads": 2,
            "d_model": 16,
            "d_ff": 32,
        },
        "training": {"epochs": 1, "batch_size": 2, "device": "cpu"},
    }
    exported = client.post(
        "/api/config/export",
        json={"name": "api-config", "spec": spec},
    )
    assert exported.status_code == 200, exported.text
    yaml_text = exported.json()["yaml"]
    assert "api-config" in yaml_text
    imported = client.post("/api/config/import", json={"yaml": yaml_text})
    assert imported.status_code == 200
    assert imported.json()["config"]["name"] == "api-config"


def test_estimate_endpoints_and_training_guard() -> None:
    huge = {
        "model": {
            "max_seq_len": 32768,
            "n_layers": 24,
            "n_heads": 16,
            "d_model": 1024,
            "d_ff": 4096,
        },
        "training": {"batch_size": 64, "epochs": 1, "device": "cpu"},
    }
    estimate = client.post("/api/estimate/training", json=huge)
    assert estimate.status_code == 200
    assert estimate.json()["level"] == "block"
    start = client.post("/api/train/start", json=huge)
    assert start.status_code == 422
    assert client.post("/api/estimate/generate", json={"count": 1}).status_code == 200


def test_training_routes_report_errors() -> None:
    assert client.get("/api/train/status/not-a-job").status_code == 404
    assert client.post("/api/analyze/not-a-job").status_code == 422
    assert client.post("/api/predict/not-a-job", json={"count": 1}).status_code == 422
    invalid = {"data": {}, "model": {"n_layers": 0}, "training": {}}
    assert client.post("/api/train/start", json=invalid).status_code == 422


def test_generate_reports_invalid_spec_gracefully() -> None:
    payload = {
        "structure": "sphere",
        "n_points": 40,
        "ambient_dim": 2,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422
    assert "ambient" in response.json()["detail"].lower()


def test_tokenize_endpoint() -> None:
    response = client.post("/api/tokenize", json={"text": "1.0;2.0<endoftext>1,2,1"})
    assert response.status_code == 200
    body = response.json()
    assert body["tokens"][0] == "1"
    assert "<endoftext>" in body["tokens"]
    assert len(body["ids"]) == len(body["tokens"])


def test_tokenize_reports_unknown_character() -> None:
    response = client.post("/api/tokenize", json={"text": "1.0?"})
    assert response.status_code == 422


def test_target_parse_endpoint() -> None:
    response = client.post(
        "/api/targets/parse",
        json={
            "text": "1,2,1;1,2,1;1;1;0,2,0.5,2",
            "schema": {
                "blocks": [
                    {"name": "betti", "fields": ["b0", "b1", "b2"]},
                    {"name": "ph", "fields": ["b0", "b1", "b2"]},
                    {"name": "genus", "fields": ["genus"]},
                    {"name": "n_components", "fields": ["n_components"]},
                    {
                        "name": "filtration",
                        "fields": ["method_code", "maxdim", "epsilon", "field_code"],
                    },
                ]
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["target_text"] == "1,2,1;1,2,1;1;1;0,2,0.5,2"


def test_target_parse_reports_shape_error() -> None:
    response = client.post("/api/targets/parse", json={"text": "1,2,1;1,2,1"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["expected"] == 1
