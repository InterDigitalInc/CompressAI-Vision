import os
from pathlib import Path


def loadEncoderDecoderFromPath(compression_model_path):
    # compression from a custcom compression model
    model_file = Path(compression_model_path) / "model.py"
    if model_file.is_file():
        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location("module", model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(
                "loading model from directory",
                compression_model_path,
                "failed with",
                e,
            )
            raise
        else:
            assert hasattr(
                module, "getEncoderDecoder"
            ), "your module is missing getEncoderDecoder function"
            encoder_decoder_func = (
                module.getEncoderDecoder
            )  # a function that returns EncoderDecoder instance
            print("loaded custom model.py")
    else:
        raise FileNotFoundError(f"No model.py in {compression_model_path}")
    return encoder_decoder_func


def setupVTM(p):
    if p.vtm_dir is None:
        try:
            vtm_dir = os.environ["VTM_DIR"]
        except KeyError as e:
            print("please define --vtm_dir or set environmental variable VTM_DIR")
            raise e
    else:
        vtm_dir = p.vtm_dir

    vtm_dir = os.path.expanduser(vtm_dir)

    if p.vtm_cfg is None:
        # vtm_cfg = getDataFile("encoder_intra_vtm_1.cfg")
        # print("WARNING: using VTM default config file", vtm_cfg)
        raise BaseException("VTM config is not defined")
    else:
        vtm_cfg = p.vtm_cfg
    vtm_cfg = os.path.expanduser(vtm_cfg)  # some more systematic way of doing these..
    print("Reading vtm config from: " + vtm_cfg)
    assert os.path.isfile(vtm_cfg), "vtm config file not found"
    # try both filenames..
    vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStatic")
    if not os.path.isfile(vtm_encoder_app):
        vtm_encoder_app = os.path.join(vtm_dir, "EncoderAppStaticd")
    if not os.path.isfile(vtm_encoder_app):
        raise AssertionError("FATAL: can't find EncoderAppStatic(d) in " + vtm_dir)
    # try both filenames..
    vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStatic")
    if not os.path.isfile(vtm_decoder_app):
        vtm_decoder_app = os.path.join(vtm_dir, "DecoderAppStaticd")
    if not os.path.isfile(vtm_decoder_app):
        raise AssertionError("FATAL: can't find DecoderAppStatic(d) in " + vtm_dir)

    return vtm_encoder_app, vtm_decoder_app, vtm_cfg
