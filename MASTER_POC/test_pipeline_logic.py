
import unittest
import os
import json
import shutil
import tempfile
import numpy as np
import onnx
from onnx import helper, TensorProto
from grok_inference_split_model_pipeline import SequentialModelPipeline
from unittest.mock import MagicMock, patch

class TestPipelineLogic(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, "test_model")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create dummy model_info.json
        self.model_info = {
            "num_layers": 1,
            "components": {
                "layer_0": {
                    "file": "layer_0.onnx",
                    "input_names": ["input_ids", "attention_mask"],
                    "output_names": ["hidden_states", "present.0.key", "present.0.value"]
                }
            },
            "original_model": "dummy",
            "architecture": "llama"
        }
        with open(os.path.join(self.model_dir, "model_info.json"), 'w') as f:
            json.dump(self.model_info, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('grok_inference_split_model_pipeline.AutoTokenizer')
    @patch('grok_inference_split_model_pipeline.load_model_config')
    @patch('grok_inference_split_model_pipeline.KVCacheManager')
    def test_detect_kv_format(self, mock_kv, mock_config, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_config.return_value = MagicMock(num_layers=1, num_kv_heads=1, head_dim=32, dtype="float32")
        
        pipeline = SequentialModelPipeline(self.model_dir, tokenizer_path="dummy", verbose=False)
        
        # Test 1: Standard format
        k, v = pipeline._detect_kv_format()
        self.assertEqual(k, "present.{layer}.key")
        self.assertEqual(v, "present.{layer}.value")
        
        # Test 2: Custom format
        pipeline.model_info["components"]["layer_0"]["output_names"] = ["out", "key_states_0", "value_states_0"]
        k, v = pipeline._detect_kv_format()
        self.assertEqual(k, "key_states_{layer}")
        self.assertEqual(v, "value_states_{layer}")

    @patch('grok_inference_split_model_pipeline.AutoTokenizer')
    @patch('grok_inference_split_model_pipeline.load_model_config')
    @patch('grok_inference_split_model_pipeline.KVCacheManager')
    def test_missing_input_raises_error(self, mock_kv, mock_config, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        pipeline = SequentialModelPipeline(self.model_dir, tokenizer_path="dummy", verbose=False)
        
        # Mock session
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "required_input"
        mock_input.type = "tensor(float)"
        mock_session.get_inputs.return_value = [mock_input]
        
        # Update model info to require this input
        pipeline.model_info["components"]["test_comp"] = {
            "input_names": ["required_input"], 
            "output_names": []
        }
        
        # Verify RuntimeError is raised
        with self.assertRaisesRegex(RuntimeError, "Missing inputs"):
            pipeline.run_component_session("test_comp", mock_session)

if __name__ == '__main__':
    unittest.main()
