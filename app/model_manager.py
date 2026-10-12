import os
import threading



class ModelManager:

    def __init__(self):

        self.models = {}

        self.lock = threading.Lock()

    def get_model(
        self,
        model_name,
        model_version,
        model_path
    ):

        key = (
            f"{model_name}:{model_version}"
        )

        if key in self.models:
            return self.models[key]

        with self.lock:

            if key in self.models:
                return self.models[key]

            if not os.path.exists(
                model_path
            ):
                raise FileNotFoundError(
                    model_path
                )

            print(
                f"Loading model "
                f"{key} from "
                f"{model_path}",
                flush=True
            )

            # Import lười để API có thể khởi động mà không phải nạp
            # toàn bộ Ultralytics/PyTorch. GPU worker vẫn cài dependency này.
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "Suy luận YOLO trực tiếp không khả dụng trong API nhẹ; "
                    "hãy gửi tác vụ qua gpu-worker."
                ) from error

            model = YOLO(
                model_path
            )

            self.models[key] = model

            return model


model_manager = ModelManager()
