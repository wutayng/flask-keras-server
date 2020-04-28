# Flask Serving Application for Keras Models

## Specially UNet-2DPose Models from [2d-pose-estimation-tf-keras](https://github.com/wutayng/2d-pose-estimation-tf-keras)

### Example Output Heatmaps

<div><img width="450" src="https://github.com/wutayng/2d-pose-estimation-tf-keras/blob/master/assets/heatmap-inference.png" />
<img width="400" src="https://github.com/wutayng/2d-pose-estimation-tf-keras/blob/master/assets/beach_running_inference.gif" />
</div>

## Start Flask API Server (Local)

```
source venv/bin/activate
```

```
python3 app.py
```

##API Methods

### Get Heatmap Video from Model Inference

```
curl -F 'file=@/path/to/input.mp4' http://localhost:5000/api/v1.0/predictVideo --output '/path/to/output.mp4'
```
