<h1>大作业一:2048人工智障模式</h1>
<text>
  提示模式见2文件夹内2048_pygame.py 写的很乱，但是算是我自己研究的一种启发模式<br>
大体思路如下:<br>
  1.最大数在四角优先，若只有一个返回该方向，若有多个执行下一步<br>
  2.比较第一步筛选的方向中，谁的最大数最大，若只有一个返回该方向，若有多个执行下一步<br>
  3.在第二步筛选基础上,看看谁的矩阵中剩下的0数目最少，若只有一个返回该方向，若有多个执行(我觉得不会这么巧吧，应该没了)<br>
这个算法还是在只考虑当前下一步最优解还是有一定的效果<br>
</text>
<text>
  人工智障模式见2048_new 游戏界面和AI界面分隔开，清晰易懂,代码源自(https://github.com/ovolve/2048-AI)<br>
  main.py中定义pygame的主体游戏进程类<br>
  functinal.py中定义矩阵类mat，其包含多种方法用于支持AI类进行判断和计算<br>
  ai.py中定义ai类，其包含Alpha-beta剪枝的Minimax的search算法，其主要思想是递归调用<br>
</text>

  
