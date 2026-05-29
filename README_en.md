# LangGrasp-Piper

LangGrasp-Piper is a language-guided grasping experimental project for the Piper robotic arm. The project integrates a MuJoCo simulation, D435i RGB-D camera, YOLO-World open-vocabulary detection, FastSAM segmentation, AnyGrasp grasp pose generation, inverse kinematics, and optional speech input into a single pipeline, allowing users to specify target objects via text or voice and perform grasping and placing in simulation.

![Project Flowchart](docs/mind_map.png)

The project is currently research and experimental in nature, suitable for learning about robot perception, language grounding, point cloud grasping, MuJoCo robotic arm control, and related directions.

## Project Features

- **Language-Guided Grasping**: Input target text, e.g., `red cube`, `bottle`, `杯子` (cup); the system attempts to detect and grasp the corresponding object.
- **Open-Vocabulary Detection**: Uses YOLO-World to locate targets in RGB images based on text prompts.
- **Automatic/Manual Segmentation**: Uses FastSAM to automatically generate masks; also supports manually selecting target areas via the Matplotlib interface.
- **RGB-D Point Cloud Generation**: Renders RGB and depth images from the D435i camera in MuJoCo and converts them to a local point cloud.
- **AnyGrasp Grasp Candidates**: Generates grasp poses based on the target point cloud, selects the best grasp result for execution.
- **Piper Robotic Arm Control**: Includes MuJoCo scene, Piper model, D435i model, gripper control, and basic motion sequences.
- **Inverse Kinematics Solver**: Defaults to MuJoCo/Scipy-based IK, with IKPy backend as an optional alternative.
- **Optional Speech Input**: Supports PyAudio recording, Vosk offline recognition, and Tencent Cloud ASR online recognition.
- **No-Voice Mode**: `main_wo_voice.py` supports pure text input, convenient for debugging vision and grasping pipelines.
- **Logging**: Integrates NoPrint logging tool, outputs `.log` and `.jsonl` files for playback and troubleshooting.

## Project Structure

```text
.
├── main.py                         # Voice/text full pipeline entry point
├── main_wo_voice.py                # Text-only pipeline entry point (no voice)
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata and command entry
├── script/
│   ├── setup.bash                  # No environment creation: installs dependencies and checks/downloads models
│   └── setup_wo_env.bash           # Creates/activates conda environment, then calls setup.bash
├── config/
│   ├── d435i_camera_params.yaml    # D435i camera parameters
│   ├── request.json                # Tencent Cloud ASR request configuration
│   └── TX-cloud_API.yaml.example   # Tencent Cloud API key template
├── model/
│   ├── FastSAM-x.pt                # FastSAM weights (auto-downloaded by script)
│   ├── yolov8x-worldv2.pt          # YOLO-World weights (auto-downloaded by script)
│   └── vosk-model-*                # Vosk offline speech model
├── piper_d435i/                    # MuJoCo Piper + D435i scene and model assets
├── src/
│   ├── Vision/                     # Detection, segmentation, point cloud, AnyGrasp workflow
│   ├── Voice/                      # Recording, offline recognition, Tencent Cloud online recognition
│   ├── ik/                         # Piper inverse kinematics controller
│   ├── NoPrint/                    # Logging submodule
│   └── grasp_sequence.py           # Grasping/placing action sequence
├── third_party/
│   └── anygrasp_sdk/               # AnyGrasp SDK submodule
└── temp/                           # Runtime logs, images, grasp results, etc.
```

## Installation

Linux + Conda is recommended. The project depends on MuJoCo, Open3D, Torch, OpenCV, PyAudio, etc. Installing these directly into the system Python can easily cause environment conflicts.

### 1. Clone the Repository

**bash**

```
git clone --recursive https://github.com/LiO2-coder/LangGrasp-Piper.git
cd LangGrasp-Piper
```

If you cloned without submodules:

**bash**

```
git submodule update --init --recursive
```

### 2. One-Click Environment Creation and Installation

**bash**

```
./script/setup_wo_env.bash
```

By default, this creates a conda environment named `mujoco` using Python `3.10`:

**bash**

```
ENV_NAME=mujoco PYTHON_VERSION=3.10 ./script/setup_wo_env.bash
```

### 3. Installation in an Existing Environment

If you have manually created and activated an environment, you can simply run:

**bash**

```
conda activate mujoco
./script/setup.bash
```

`script/setup.bash` does three things:

* Installs `requirements.txt`
* Installs `graspnetAPI==1.2.11` with `--no-deps`
* Checks for and downloads `model/FastSAM-x.pt` and `model/yolov8x-worldv2.pt`

### 4. Configure AnyGrasp

Running AnyGrasp requires a license due to copyright issues. You need to fill out a form with your machine code. After about 2 days, you will receive a reply email containing the registration file and weight download link for your machine.

For details, refer to [Anygrasp 2025 Configuration Guide](https://zhuanlan.zhihu.com/p/1924881466229233373).

### 5. Configure Tencent Cloud ASR (Optional)

To use online speech recognition, please refer to the [Tencent Cloud API Documentation](https://console.cloud.tencent.com/api/explorer?Product=asr&Version=2019-06-14&Action=CreateRecTask).
Copy the template and fill in your own API keys:

**bash**

```
cp config/TX-cloud_API.yaml.example config/TX-cloud_API.yaml
```

Alternatively, use environment variables:

**bash**

```
export TENCENTCLOUD_SECRET_ID="Your SecretId"
export TENCENTCLOUD_SECRET_KEY="Your SecretKey"
export TENCENTCLOUD_REGION="ap-guangzhou"
```

Do **not** commit your actual `config/TX-cloud_API.yaml` to Git; cloud services cost money!

## Dependencies

Main Python dependencies include:

* `mujoco`: MuJoCo physics simulation and rendering
* `numpy` / `scipy`: Numerical computation, inverse kinematics optimization
* `opencv-python` / `Pillow`: Image processing
* `matplotlib`: Interactive main control interface
* `open3d`: Point cloud processing and visualization
* `ultralytics`: YOLO-World and FastSAM inference
* `torch`: Deep learning inference backend
* `PyAudio` / `vosk` / `tencentcloud-sdk-python-asr`: Speech input and recognition
* `ikpy`: Optional IK backend
* `graspnetAPI`: AnyGrasp result structure support

Note: The `graspnetAPI==1.2.11` package metadata forces `numpy==1.20.3`, which conflicts with the project's modern dependency stack. Therefore, it is not included in `requirements.txt` but is installed separately in `script/setup.bash` using `pip install --no-deps`.

System-side recommendations:

* Conda / Miniconda
* Functional OpenGL/GLFW environment
* Microphone device (only required for voice mode)
* NVIDIA GPU with CUDA (recommended, but not mandatory for all debugging steps)

## Quick Start

### Text Input Mode

**bash**

```
conda activate mujoco
python main_wo_voice.py
```

After launching, enter the target text in the Matplotlib interface. Confirm detection, segmentation, and grasp results, then execute the grasping action.

### Voice Input Mode

**bash**

```
conda activate mujoco
python main.py
```

Click `Record` to start recording; click again to stop. The system will prioritize Tencent Cloud ASR; if the network is unavailable or online recognition fails, it will fall back to Vosk offline recognition.

### Demo Screenshots

![MuJoCo](docs/scene.png)

![GUI](docs/MuJoCo_Grasp_Workflow.png)

## Notes

* The model weights `FastSAM-x.pt` and `yolov8x-worldv2.pt` are large and are not recommended to be committed directly to Git; the installation script downloads them if missing.
* Logs, images, JSON results, point cloud files, etc., generated during runtime are written to `temp/`.
* `config/TX-cloud_API.yaml` is a local key file and has been ignored by `.gitignore`.
* The AnyGrasp SDK is a submodule located at `third_party/anygrasp_sdk`. If your runtime configuration still points to the adjacent directory `../anygrasp_sdk/grasp_detection`, please ensure the local path is consistent or change the code configuration to `third_party/anygrasp_sdk/grasp_detection`.
* The project is primarily intended for simulation environments. Deployment on a real robotic arm requires additional calibration, safety limits, collision detection, emergency stops, and hardware communication.
* MuJoCo rendering depends on the local graphics environment; when running on a headless server, you may need to configure EGL, a virtual display, or remote graphics forwarding.

## TODO: Development Plan

* Integrate the pressure sensor already on the Piper gripper to add gripping force control.
* Develop a more user-friendly GUI that unifies detection, masks, point clouds, and grasp candidates in one workspace.
* Support multi-object tasks, e.g., "put the red cube into the blue box."
* Add automatic failure recovery: retry on detection failure, switch grasp candidates on failure, replan on IK failure.
* Add rigorous collision detection and trajectory smoothing.
* Sim2Real: interface with the real Piper robotic arm, migrating the simulation pipeline to the physical robot.

## Contributing

Issues, PRs, and experiment logs are welcome. Recommended ways to contribute:

* When reporting installation issues, include system version, Python version, CUDA/Torch version, and full error messages.
* When modifying vision or grasping pipelines, include a set of reproducible experiment screenshots or logs.
* Do not commit keys, temporary logs, large model files, `__pycache__`, or local virtual environments.
* If adding third-party models or SDKs, please indicate the source, license, and download method.

## Acknowledgements

This project thanks many educational bloggers and excellent open-source projects, including:

* MuJoCo tutorial blogger [材机战士](https://github.com/Albusgive) on Bilibili
* AnyGrasp tutorial blogger [忘中犹记ZihaoLiu](https://www.zhihu.com/people/clsr-44) on Zhihu
* Inspiration from Bilibili blogger [猪猪呀咋](https://space.bilibili.com/3632305260202417?spm_id_from=333.788.upinfo.detail.click)
* MuJoCo: Physics simulation and robot modeling
* Ultralytics / YOLO-World: Open-vocabulary object detection
* FastSAM: Fast segmentation model
* AnyGrasp / GraspNet: Grasp pose generation and grasping data ecosystem
* Open3D: Point cloud processing and visualization
* Matplotlib: Interactive debugging interface
* Vosk: Offline speech recognition
* Tencent Cloud ASR: Online speech recognition
* NoPrint: Logging utility used in this project
* AgileX Piper and RealSense D435i related model assets

## License

This project is licensed under the MIT License - see the [LICENSE](https://license/) file for details.

Third-party models, SDKs, pre-trained weights, robotic assets, and speech services are subject to their original projects or terms of service. Before use, distribution, or commercial application, please confirm the corresponding licenses.

## Author

* **GitHub**: [https://github.com/LiO2-coder](https://github.com/LiO2-coder)

Project Name: LangGrasp-Piper

Direction: Language-guided robotic grasping, simulation perception, robotic learning toolchain

## Version History

### v1.0

* Built MuJoCo Piper + D435i simulation scene.
* Integrated YOLO-World, FastSAM, AnyGrasp.
* Added Matplotlib interactive grasping pipeline.
* Added text input mode and voice input mode.
* Added installation scripts, dependency files, and automatic model download logic.
* Added basic logging and runtime result saving.

## Support

If execution fails, first check this information:

**bash**

```
python --version
python -m pip list | grep -E "mujoco|torch|ultralytics|open3d|numpy|scipy|graspnet"
git submodule status
ls -lh model/
ls -lh temp/log/
```

Common issues include:

* `graspnetAPI` and `numpy` dependency conflict: Use `script/setup.bash`, do not let pip resolve dependencies for `graspnetAPI` manually.
* Model weights not found: Ensure `model/FastSAM-x.pt` and `model/yolov8x-worldv2.pt` exist.
* AnyGrasp checkpoint not found: Ensure the AnyGrasp SDK submodule has been pulled and check the SDK path in the code configuration.
* PyAudio installation failure: Prefer using conda to install `portaudio`, then install Python dependencies.
* Tencent Cloud ASR failure: Check API keys, region, network; you can first use the no-voice mode to debug the vision pipeline.

When seeking help, please include the full error message, execution command, system environment, and the path to the latest log file. Logs are usually located in `temp/log/`.

You can reach out via:

* GitHub Issues: [Project Issues Page](https://github.com/LiO2-coder/EasyFilter/issues)

---

⭐ If this project helps you, please give it a Star!
