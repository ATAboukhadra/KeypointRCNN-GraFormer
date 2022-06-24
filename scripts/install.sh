conda create -n pytorch3d python=3.8
conda activate pytorch3d
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt


