---
id: b3xypywbus6o3g6k785m5lm
title: 工程能力
desc: ''
updated: 1744815880912
created: 1740223986543
---

加强工程能力，把算法、想法用代码快速实现的能力。需要多阅读，多实现。关注两个方面：
- **理论基础要扎实**，要在工程问题中灵活运用，以洞察问题关键。需要掌握基础知识，常用的算法，经典的论文、博客和文章。多阅读思考，窥探细节，加深印象，提出见解。
- **掌握常用的工具**。比如用好常用编程语言，能够有阅读和梳理工程项目。这需要掌握方法和技巧，比如阅读时，关注数据流；再比如，从文档了解各个模块之间关系。通过多尝试 LeetCode，开源项目，提升编程能力，能够正确实现，快速实现。
- **加强抽象工程问题和解决问题的能力**，把问题抽象为数字、模型等抽象概念，再用工具尝试解决。
- **关注效率**。边干边学，而不是学了再干。比如，某些工具的各类参数，在实践中去查阅和了解，而不是一蹴而就。
- **梳理知识、总结经验**。通过梳理和总结，加深对知识、对项目理解。抽丝剥茧时，往往会发现各个技术点之间的联系，可以排查低效和冗余，有时候创新便是对这些联系的改进。及时整理或安排时间整理笔记和总结经验，是高效学习的方法。
- **整理工作流程，形成制度**。约束能够促进规范操作，对着条目逐步执行和排查，可以避免忘记细节和遗漏。特别是琐碎的工作和细节，很难全部记忆，整理称为工作流程后，对照执行能够避免记忆和梳理负担。

## 关注效率

合理规划时间。时间比较重要，特别是有任务和目的时，要科学制定下一步工作。比如，目前需要复现 DexVLA，需要用在机器人上。梳理脉络后，比如，我想要再探究 Fusion 模块是如何融合大模型隐藏状态和文本到 action_hidden_states的，但这可能过于刨根问底。另一个选择，先有大致了解即可，知道它用了什么方法。

先有全局认识，先梳理完脉络，高屋建瓴。运行起来，再逐个分析，逐个模块学习。时间成本可能较贵，需要合理安排，有时只用了解输入输出的规范，先当做黑盒，知道功能即可。后续再挑选合适的模块学习。当务之急是先打通所有模块，把各个模块接口的通信方式，数据路径梳理清楚。

看项目用到什么参数，就查阅参数用法，作笔记记录。

## 关于代码能力

代码是工具，需要设计和实现。提升设计和实现能力，需要多读、多写。也就需要累积足够多的经验，学透领域常用知识，熟练使用常用工具，熟悉常见编程范式。

学习需要**模仿和借鉴，大量实践**。模仿优秀的代码，借鉴优秀的设计。才能有深刻印象，有感性认识。甚至需要找一个项目，自己敲一遍代码，不要复制粘贴。这样才能加深印象，让自己熟悉设计的思路，编程的细节。

多做总结和思考。总结设计思路，总结实现细节，总结如何 debug。

## 关注解决问题

首先，精准定位问题。比如，运行时报错，分析是哪个模块的问题，定位到具体模块，函数。紧接着，对问题分类，是数据格式不对，还是参数不对。

其次，分析问题的原因。找到引起问题的导火索。比如，数据格式不对，是因为数据预处理时，数据格式不对，还是模型输入时，数据格式不对。这需要了解各个模块的结构，特别是输入输出数据格式，数据流向，才能定位原因。

设计解决方案，比如，数据格式不对，可能是数据预处理时，数据格式不对。可以尝试修改数据预处理的代码，或者修改模型输入的代码。

如果无法解决问题，要能够提供有效信息，有思考和尝试。向团队协调资源，提出意见，积极沟通寻找新的方案。

## 关注总结和整理

总结经验和写记录时，尽量使用总分总结构，文章会更易懂。使用分总，容易在顺着读时，对全盘认识不够，不了解此工作的影响，容易忽略至关重要的细节，让人难以抓重点。比如，查看 CLIPVisionModel 时，应当先总结，ViT 部分包含了哪些组件，再一一分析，最后再总结作用和特点。

在总结和整理中，可以关注如下方面：
1. 知识树（图）。任何知识，只在点上学习不够的，需要在面上学习，这叫系统地学习，这需要我们去总结并归纳知识树或知识图，一个知识面会有多个知识板块组成，一个板块又有各种知识点，一个知识点会导出另外的知识点，各种知识点又会交叉和依赖起来，学习就是要系统地学习整个知识树（图）。而我们都知道，对于一棵树来说，“根基”是非常重要的，所以，学好基础知识也是非常重要的，对于一个陌生的地方，有一份地图是非常重要的，没有地图的你只会乱窜，只会迷路、练路、走冤枉路！
2. **知识缘由**。任何知识都是有缘由的，**了解一个知识的来龙去脉和前世今生，会让你对这个知识有非常强的掌握**，而不再只是靠记忆去学习。靠记忆去学习是一件非常糟糕的事。而对于一些操作性的知识（不需要了解由来的），我把其叫操作知识，就像一些函数库一样，这样的知识只要学会查文档就好了。能够知其然，知其所以然的人自然会比识知识到表皮的人段位要高很多。
3. 方法套路。学习不是为了找到答案，而是找到方法。就像数学一样，你学的是方法，是解题思路，是套路，会用方程式解题的和不会用方程式解题的在解题效率上不可比较，而在微积分面前，其它的解题方法都变成了渣渣。你可以看到，掌握高级方法的人比别人的优势有多大，学习的目的就是为了掌握更为高级的方法和解题思路。
4. 坚持和勤快。

学习技能要**精益求精**。如果你想拥有专业的技能，不仅仅是拼命地重复一遍又一遍的训练，而是在每一次重复训练时你都要**找到更好的方法，总结经验**，让新的一遍能够更好，更漂亮，更有效率，否则，用相同的方法重复，那你只不过在搬砖罢了。

https://coolshell.cn/articles/19464.html

### 教程类和经验类文字

先写出总体的和概览的概念解释，随后再引入一个简单直观的样例（参数、命名直观，用法新内容负担最小），随后再展开详细解释，最后总结。这样的文章在逻辑理解上更加平滑。