# InstrcutGPT论文—DeepSpeed开源代码

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/DEoZWY9NU5a4GN1o/f31f5405389b436fa8d949c3e5c68d010628.png)

[《DeepSpeed code .adoc》](https://alidocs.dingtalk.com/document/edit?dentryKey=AJEyjQmkf3lPDKEG&type=d&utm_medium=drive_myfile&utm_source=drive#)

【三阶段分析】00:22:18 - 00:26:50

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/DEoZWY9NU5a4GN1o/f9ee451bb3754288a56733fe98d3efed0628.png)

【SFT模型】00:41:33 

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/DEoZWY9NU5a4GN1o/db6f7dc65d494ad59e76fed731ed4df80628.png)

【RM模型】00:42:07 - 

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/DEoZWY9NU5a4GN1o/4b79aa3ea2b144a3af32304f32b8790f0628.png)

【PPO模型】00:49:44 - 00:58:00

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/DEoZWY9NU5a4GN1o/58d05b6ea3084512807626b8cf14995d0628.png)

【实验结果】01:00:00

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/DEoZWY9NU5a4GN1o/d847845410344a7fa948fde1fc2795770628.png)

Reference:

1.  Training language models to follow instructions with human feedback, 2022, 03; 
    
2.  Proximal Policy Optimization Algorithms, 2017, 08;
    
3.  B 站 李沐论文分享 [https://www.bilibili.com/video/BV1hd4y187CR/?spm\_id\_from=333.337.search-card.all.click&vd\_source=3157f49e1a4d8799d5ac7e5ea94c4bb9](https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.337.search-card.all.click&vd_source=3157f49e1a4d8799d5ac7e5ea94c4bb9)
    
4.  DeepSpeedExamples开源代码 [https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training)
    

Q&A:

1.  Actor Model和Reward Model大小和参数量是否需要一致，即6B SFT是否必然与6B Reward Model对应。
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/DEoZWY9NU5a4GN1o/6a64ee89035f4493bfc30d296aa931530628.png)

出自Instruct GPT原文附录C.2。