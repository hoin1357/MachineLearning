from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.model_runtime import PortablePredictionRuntime  # noqa: E402


def main() -> None:
    runtime = PortablePredictionRuntime()
    runtime.initialize()
    sample_date = runtime.min_supported_date.isoformat()
    sample_prediction = runtime.predict_for_date(runtime.min_supported_date)
    print("artifacts_ready", runtime.health())
    print("sample_date", sample_date)
    print("sample_prediction", sample_prediction["predictedVisitors"], sample_prediction["congestionLevel"])


if __name__ == "__main__":
    main()
