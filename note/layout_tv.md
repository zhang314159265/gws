```
thr_layout = (T1, T2) : (d1, d2)
val_layout = (V1, V2) : (d3, d4)

T = T1 * T2
M = T1 * V1
N = T2 * V2

raked_product_layout = layout_mn = ((V1, T1), (V2, T2)) : ((d3 * T, d1), (d4 * T, d2))

given m, n
coord = (m % V1, m // V1), (n % V2, n // V2)
idx = m % V1 * d3 * T + m // V1 * d1 + n % V2 * d4 * T + n // V2 * d2

t = m // V1 * d1 + n // V2 * d2, (we have t < T)
v = m % V1 * d3  + n % V2 * d4

(the following 4 lines assumes thr_layout/val_layout are compact. Compact layout guarantees unique solution for t1,t2,v1,v2)
t1 = m // V1
t2 = n // V2
v1 = m % V1
v2 = n % V2

This result shows that TV layout replicate the val_layout by thr_layout!

```
