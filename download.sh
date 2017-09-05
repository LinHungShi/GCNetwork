mkdir -p data2/sceneflow/driving &&
mkdir -p data2/sceneflow/monkaa &&
cd data2/sceneflow/driving &&
wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_finalpass.tar && tar -xzf driving__frames_finalpass.tar &&
wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/derived_data/driving__disparity.tar.bz2 && tar -xzf driving__disparity.tar.bz2 &&
cd .. && cd monkaa &&
wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_cleanpass.tar && tar -xzf monkaa__frames_cleanpass.tar &&
wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__disparity.tar.bz2 && tar -xzf monkaa__disparity.tar.bz2
cd .. && cd .. && cd ..
