<h1>大作业一:2048人工智障模式</h1>
提示模式见2文件夹内2048_pygame.py 写的很乱，但是算是我自己研究的一种启发模式
大体思路如下:
  1.最大数在四角优先，若只有一个返回该方向，若有多个执行下一步
  2.比较第一步筛选的方向中，谁的最大数最大，若只有一个返回该方向，若有多个执行下一步
  3.在第二步筛选基础上,看看谁的矩阵中剩下的0数目最少，若只有一个返回该方向，若有多个执行(我觉得不会这么巧吧，应该没了)
这个算法还是在只考虑当前下一步最优解还是有一定的效果
  
