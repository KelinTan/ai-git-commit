# ai-git-commit
最简单的 ai-git-commit 生成脚本，就一个文件。

# 前言
偶然间看到这个项目 https://github.com/Nutlope/aicommits 用起来很 nice，苦于 openai 必须要充值，白嫖党决定自己实现来接入国内的 ai 厂商，将白嫖进行到底，接入了多个平台，方便快速切换。


# 使用
很简单，调用即可。

> 注意环境变量设置不同平台的KEY,如阿里的DASHSCOPE_API_KEY、智谱的ZHIPU_API_KEY等等，瞅下代码

自定义了一个alias来快速使用：

![image](https://github.com/KelinTan/ai-git-commit/assets/23694073/f122013f-502b-4853-9fcc-6a82c7d422d6)


使用如截图：

![image](https://github.com/KelinTan/ai-git-commit/assets/23694073/41341eb0-598a-4445-af9f-ce033809dda6)


# 一些配置
![image](https://github.com/KelinTan/ai-git-commit/assets/23694073/059690c2-fb65-4084-b69d-587db9b8ce6f)

--model 切换不同平台，支持baidu、zhipu、ali、openai

--style 切换 git 风格，普通的style或git-commit规范

# 不同平台使用心得（~~白嫖心得~~ ）

| 平台      | 评价 |
| ----------- | ----------- |
|  openai      | gpt4效果好但太贵🥰，gpt3.5 😂       |
|  zhipu   | 4.0效果好但太贵🥰，3.0😂        |
|  baidu   | 🙄        |
|  ali   | 平替，正在用😄        |

# 一些小问题（~~不想修~~）

- 上下文长度问题：如果缓存区一次性代码改动太多就会超过模型的上下文长度。一般建议拆分commit
- commit不符合要求：肯定是模型的问题，不是我的问题
- 想要一次性多个候选commit: 太麻烦，建议试多次😂 


