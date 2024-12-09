clc, clear;
figure(3)

% 生成 x 和 z 网格数据
[x, z] = meshgrid(0:0.01:1, 0:0.01:1);

% 参数定义
rg = 100; rm = 100; rN = 100;
a = 0.15; a1 = 0.2; a2 = 0.3; 
Lg0 = 2.6/526.9175; Lg1 = 10;
Cg = 1528.5; zeta = 505; Rt = 38; RL = 13;
Cm = 320.5; Lm = 10; Dt = 34;
Br = 10; Lr = 10; Ba = 10; La = 10;
eta1 = 0.7; eta2 = 0.72; delta1 = 0.25; delta2 = 0.3;

% 计算 y 的值
L = rm*a*(a2-2*a1);
P = rm*a*(a1-a2);
Q = rm*(a2-2*a1);
R = 8.5;
y = -(P*x + R) ./ (L*x + Q);  % 点运算

% 绘制图像
mesh(x, y, z,'FaceColor', [227/255, 115/255, 139/255], 'EdgeColor', [227/255, 115/255, 139/255])
colormap(winter) 
alpha(0.0001)
view([135 25]);
set(gca, 'XTick', [0:0.5:1], 'YTick', [0:0.2:1], 'ZTick', [0:0.1:1])
axis([0 1 0 1 0 1])
xlabel('$x$', 'interpreter', 'latex', 'FontSize', 18);
ylabel('$y$', 'interpreter', 'latex', 'FontSize', 18);
zlabel('$z$', 'interpreter', 'latex', 'Rotation', 360, 'FontSize', 18);
grid on
hold on
set(0, 'defaultfigurecolor', 'w')
text(0.5, 0, 0.8, '$(a)y=y^{*}$', 'interpreter', 'latex', 'FontSize', 18);
text(0.4, 0, 0.8, '$(b)y<y^{*},x\rightarrow0$', 'interpreter', 'latex', 'FontSize', 18);
text(0.8, 0, 0.8, '$(c)y>y^{*},x\rightarrow1$', 'interpreter', 'latex', 'FontSize', 18);
annotation('arrow', [0.55 0.35], [0.35 0.32]);
annotation('arrow', [0.58 0.58], [0.38 0.45]);

% 生成 x 和 z 网格数据
[x, z] = meshgrid(0:0.01:1, 0:0.01:1);

% 参数定义
rg = 100; rm = 100; rN = 100;
a = 0.15; a1 = 0.2; a2 = 0.3; 
Lg0 = 2.6/526.9175; Lg1 = 10;
Cg = 1528.5; zeta = 505; Rt = 38; RL = 13;
Cm = 320.5; Lm = 10; Dt = 34;
Br = 10; Lr = 10; Ba = 10; La = 10;
eta1 = 0.7; eta2 = 0.72; delta1 = 0.25; delta2 = 0.3;

% 计算 y 的值
L = rm*a*(a2-2*a1);
P = rm*a*(a1-a2);
Q = rm*(a2-2*a1);
R = 2;
y = -(P*x + R) ./ (L*x + Q);  % 点运算

% 绘制图像
%mesh(x, y, z,'FaceColor', [140/255, 165/255, 234/255], 'EdgeColor', [140/255, 165/255, 234/255])
% 绘制网格并使用 Z 数据作为颜色映射
mesh(x, y, z, 'FaceColor', 'interp', 'EdgeColor', 'interp');

% 使用 colormap 设置渐变颜色，或者定义自定义渐变色
colormap(jet);  % 可以使用自定义颜色映射，如 colormap([R,G,B])


% 自定义颜色渐变 (例如从蓝色到红色)
myColormap = [linspace(140/255, 227/255, 256)', ... % Red component
              linspace(165/255, 115/255, 256)', ... % Green component
              linspace(234/255, 139/255, 256)'];    % Blue component
colormap(myColormap);

colormap(winter) 
alpha(0.0001)
view([135 25]);
set(gca, 'XTick', [0:0.5:1], 'YTick', [0:0.2:1], 'ZTick', [0:0.1:1])
axis([0 1 0 1 0 1])
xlabel('$x$', 'interpreter', 'latex', 'FontSize', 18);
ylabel('$y$', 'interpreter', 'latex', 'FontSize', 18);
zlabel('$z$', 'interpreter', 'latex', 'Rotation', 360, 'FontSize', 18);
grid on
hold on
set(0, 'defaultfigurecolor', 'w')
text(0.5, 0, 0.8, '$(a)y=y^{*}$', 'interpreter', 'latex', 'FontSize', 18);
text(0.4, 0, 0.8, '$(b)y<y^{*},x\rightarrow0$', 'interpreter', 'latex', 'FontSize', 18);
text(0.8, 0, 0.8, '$(c)y>y^{*},x\rightarrow1$', 'interpreter', 'latex', 'FontSize', 18);
annotation('arrow', [0.55 0.35], [0.35 0.32]);
annotation('arrow', [0.58 3.58], [0.38 0.45]);



