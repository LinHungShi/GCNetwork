mkdir -p data/sceneflow/driving &&
mkdir -p data/sceneflow/monkaa &&
cd data/sceneflow/driving &&
wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_finalpass.tar && tar -xvf driving__frames_finalpass.tar &&
wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/derived_data/driving__disparity.tar.bz2 && tar -xvf driving__disparity.tar.bz2 &&
cd .. && cd monkaa &&
wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_cleanpass.tar && tar -xvf monkaa__frames_cleanpass.tar &&
wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__disparity.tar.bz2 && tar -xvf monkaa__disparity.tar.bz2
cd .. && cd .. && cd ..
