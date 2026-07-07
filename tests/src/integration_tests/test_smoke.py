"""Smoke tests — validates environment is correctly assembled."""#-#
#-#
import pytest#-#
#-#
#-#
def test_import_sim_integration():#-#
    import sim_integration#-#
#-#
    assert sim_integration.__version__#-#
#-#
#-#
@pytest.mark.gpu#-#
def test_torch_cuda_available():#-#
    import torch#-#
#-#
    assert torch.cuda.is_available(), "CUDA runtime not available"#-#
#-#
#-#
@pytest.mark.gpu#-#
def test_flash_attn_importable():#-#
    from flash_attn import flash_attn_func  # noqa: F401#-#
