import yaml

from open_standard_evaluation.definitions import EnrichmentDefinitions, MetricDefinition, load_definitions


class TestEnrichmentDefinitions:
    def test_empty_definitions(self):
        defs = EnrichmentDefinitions()
        assert defs.metrics == []
        assert defs.format_for_prompt() == ""

    def test_metric_lookup(self):
        defs = EnrichmentDefinitions(metrics=[
            MetricDefinition(name="tc", scale="0-1", description="Task completion"),
            MetricDefinition(name="faith", scale="1-5", description="Faithfulness"),
        ])
        assert defs.get_metric("tc").scale == "0-1"
        assert defs.get_metric("faith").scale == "1-5"
        assert defs.get_metric("missing") is None

    def test_metric_names(self):
        defs = EnrichmentDefinitions(metrics=[
            MetricDefinition(name="tc", scale="0-1", description="Task completion"),
            MetricDefinition(name="faith", scale="1-5", description="Faithfulness"),
        ])
        assert defs.metric_names == ["tc", "faith"]

    def test_format_for_prompt(self):
        defs = EnrichmentDefinitions(
            metrics=[MetricDefinition(
                name="quality",
                scale="1-5",
                description="Overall quality",
                interpretation={"5": "Perfect", "1": "Terrible"},
            )],
            custom_instructions="This is a support agent.",
        )
        text = defs.format_for_prompt()
        assert "quality" in text
        assert "1-5" in text
        assert "Perfect" in text
        assert "This is a support agent." in text


class TestLoadDefinitions:
    def test_loads_from_yaml(self, tmp_path):
        data = {
            "metrics": [
                {"name": "tc", "scale": "0-1", "description": "Task completion"},
            ],
            "noise_patterns": ["hi", "hello"],
            "custom_instructions": "Test agent",
        }
        path = tmp_path / "definitions.yaml"
        path.write_text(yaml.dump(data))

        defs = load_definitions(path)
        assert len(defs.metrics) == 1
        assert defs.metrics[0].name == "tc"
        assert defs.noise_patterns == ["hi", "hello"]
        assert defs.custom_instructions == "Test agent"

    def test_returns_empty_when_no_file(self, tmp_path):
        defs = load_definitions(tmp_path / "nonexistent.yaml")
        assert defs.metrics == []
        assert defs.noise_patterns == []
