 % Check Linear Approximation

k = 5;
x = 50;
w1 = 10;
s = 1:1:100;
real = k.*tanh((s+x+w1)./k);
appr = sech(s/k).^2*(x+w1);

close all
figure;
plot(s, real, "linewidth", 3);
hold on;
plot(s, appr, "linewidth", 3);
legend(["function", "approximation"])
xlabel("s")
title("x="+x + ", w1=" +w1 + ", k=" + k)
grid on
grid minor
