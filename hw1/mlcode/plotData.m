x= [16,  32, 64, 128, 256, 512, 1024, 2048, 4096];
x2 = [4, 5,6,7,8,9,10,11,12];
normal_LU = [0.0015, 0.0094, 0.0512, 0.3928, 3.0748,23.9196, 188.136,1664.63, 14370.2];
SSE_1 = [0.0021 ,0.008 ,0.0353,0.2822,1.3814,11.0992, 85.1287,1116.1,9653.51];
SSE_2 = [0.005,0.0087,0.0498,0.388,3.042,23.9097,188.313,1690.43,14437.9];
SSE_both = [0.002,0.0074, 0.0343,0.1967,1.2782,11.5798,83.8831,1037.25,9603.69];
plot(x, normal_LU, 'b')
hold on
plot(x, SSE_1, 'r')
hold on
plot(x, SSE_2, 'g')
hold on
plot(x, SSE_both, 'black')
grid on
xlabel('数据数量');
ylabel('耗时');
legend('普通', 'SSE_1', 'SSE_2', 'SSE_both');