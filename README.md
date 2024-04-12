基于transformer的场景文本检测的研究与改进

针对弯曲文本进行的改进
参考“CRNet：A Center-aware Representation for Detecting Text of Arbitrary Shape”改进了Backbone
使用CRNet进行特征采样，基于自注意力的transformer encoder进行特征提取，最后送入四个预测头预测，实现文本语义分割

对于弯曲文本与长文本检测效果显著

同时训练backbone和encoder，收敛迅速，在少量样本训练中表现良好
