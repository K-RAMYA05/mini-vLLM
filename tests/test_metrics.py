from mini_vllm.metrics import EngineMetrics


def test_structured_snapshot_nests_scheduler_and_graph_fields():
    metrics = EngineMetrics(
        requests_started=2,
        requests_finished=1,
        prefill_graph_hits=3,
        decode_graph_hits=4,
    )

    snap = metrics.structured_snapshot()

    assert snap["requests"]["started"] == 2
    assert snap["graphs"]["prefill_hits"] == 3
    assert snap["graphs"]["decode_hits"] == 4
