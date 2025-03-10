cd ~/piper/src/python/
python3 -m venv .venv
source .venv/bin/activate

mkdir ~/model
#you will need to rename the model file to what its actualy called
python3 -m piper_train.export_onnx \
    ~/training/lightning_logs/version_0/checkpoints/epoch=2165-step=1355546.ckpt \
    ~/model/model.onnx
    
cp ~/training/config.json \
   ~/model/model.onnx.json