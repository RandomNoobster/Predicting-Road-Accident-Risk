from pathlib import Path
import joblib
import json

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class PersistentMixin:
    def fit_or_load(self, X, y, **fit_params):
        init_params = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k != "model"
        }

        config = {
            "class": self.__class__.__name__,
            "init": init_params,
            "fit": fit_params,
        }
        unique_id = joblib.hash(config)

        base_name = f"{self.__class__.__name__}_{unique_id}"
        is_keras = hasattr(self.model, "save_weights")

        filename = f"{base_name}.weights.h5" if is_keras else f"{base_name}.pkl"
        file_path = MODEL_DIR / filename
        meta_path = MODEL_DIR / f"{base_name}.json"

        if file_path.exists():
            print(f"✅ Found cached model: {filename}")
            if is_keras:
                self.model.load_weights(file_path)
            else:
                loaded_obj = joblib.load(file_path)
                self.__dict__.update(loaded_obj.__dict__)
            return self

        print(f"⚙️  Training new model: {base_name}...")
        self.fit(X, y, **fit_params)

        if is_keras:
            self.model.save_weights(file_path)
        else:
            joblib.dump(self, file_path)

        with open(meta_path, "w") as f:
            json.dump(config, f, indent=4, default=str)

        return self