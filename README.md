# ACM_template

## 高精

### 两个非负整数相加

```c
//加法
string add(string a,string b)//只限两个非负整数相加
{
    const int L=1e5;
    string ans;
    int na[L]={0},nb[L]={0};
    int la=a.size(),lb=b.size();
    for(int i=0;i<la;i++) na[la-1-i]=a[i]-'0';
    for(int i=0;i<lb;i++) nb[lb-1-i]=b[i]-'0';
    int lmax=la>lb?la:lb;
    for(int i=0;i<lmax;i++) na[i]+=nb[i],na[i+1]+=na[i]/10,na[i]%=10;
    if(na[lmax]) lmax++;
    for(int i=lmax-1;i>=0;i--) ans+=na[i]+'0';
    return ans;
}
```

### 大的非负整数减小的非负整数

```c
//减法
string sub(string a,string b)//只限大的非负整数减小的非负整数
{
    const int L=1e5;
    string ans;
    int na[L]={0},nb[L]={0};
    int la=a.size(),lb=b.size();
    for(int i=0;i<la;i++) na[la-1-i]=a[i]-'0';
    for(int i=0;i<lb;i++) nb[lb-1-i]=b[i]-'0';
    int lmax=la>lb?la:lb;
    for(int i=0;i<lmax;i++)
    {
        na[i]-=nb[i];
        if(na[i]<0) na[i]+=10,na[i+1]--;
    }
    while(!na[--lmax]&&lmax>0)  ;lmax++;
    for(int i=lmax-1;i>=0;i--) ans+=na[i]+'0';
    return ans;
}
```

### 高精乘高精

```
string mul(string a,string b)//高精度乘法a,b,均为非负整数
{
    const int L=1e5;
    string s;
    int na[L],nb[L],nc[L],La=a.size(),Lb=b.size();//na存储被乘数，nb存储乘数，nc存储积
    fill(na,na+L,0);fill(nb,nb+L,0);fill(nc,nc+L,0);//将na,nb,nc都置为0
    for(int i=La-1;i>=0;i--) na[La-i]=a[i]-'0';//将字符串表示的大整形数转成i整形数组表示的大整形数
    for(int i=Lb-1;i>=0;i--) nb[Lb-i]=b[i]-'0';
    for(int i=1;i<=La;i++)
        for(int j=1;j<=Lb;j++)
        nc[i+j-1]+=na[i]*nb[j];//a的第i位乘以b的第j位为积的第i+j-1位（先不考虑进位）
    for(int i=1;i<=La+Lb;i++)
        nc[i+1]+=nc[i]/10,nc[i]%=10;//统一处理进位
    if(nc[La+Lb]) s+=nc[La+Lb]+'0';//判断第i+j位上的数字是不是0
    for(int i=La+Lb-1;i>=1;i--)
        s+=nc[i]+'0';//将整形数组转成字符串
    return s;
}
```

### 高精度a乘低精度b

```
string mul(string a,int b)//高精度a乘单精度b
{
    const int L=100005;
    int na[L];
    string ans;
    int La=a.size();
    fill(na,na+L,0);
    for(int i=La-1;i>=0;i--) na[La-i-1]=a[i]-'0';
    int w=0;
    for(int i=0;i<La;i++) na[i]=na[i]*b+w,w=na[i]/10,na[i]=na[i]%10;
    while(w) na[La++]=w%10,w/=10;
    La--;
    while(La>=0) ans+=na[La--]+'0';
    return ans;
}
```

### 高精除高精+高精除低精

```c

//除法
int sub(int *a,int *b,int La,int Lb)
{
    if(La<Lb) return -1;//如果a小于b，则返回-1
    if(La==Lb)
    {
        for(int i=La-1;i>=0;i--)
            if(a[i]>b[i]) break;
            else if(a[i]<b[i]) return -1;//如果a小于b，则返回-1

    }
    for(int i=0;i<La;i++)//高精度减法
    {
        a[i]-=b[i];
        if(a[i]<0) a[i]+=10,a[i+1]--;
    }
    for(int i=La-1;i>=0;i--)
        if(a[i]) return i+1;//返回差的位数
    return 0;//返回差的位数

}
string div(string n1,string n2,int nn)
//n1,n2是字符串表示的被除数，除数,nn是选择返回商还是余数
{
    const int L=1e5;
    string s,v;//s存商,v存余数
     int a[L],b[L],r[L],La=n1.size(),Lb=n2.size(),i,tp=La;
     //a，b是整形数组表示被除数，除数，tp保存被除数的长度
     fill(a,a+L,0);fill(b,b+L,0);fill(r,r+L,0);//数组元素都置为0
     for(i=La-1;i>=0;i--) a[La-1-i]=n1[i]-'0';
     for(i=Lb-1;i>=0;i--) b[Lb-1-i]=n2[i]-'0';
     if(La<Lb || (La==Lb && n1<n2)) {
            //cout<<0<<endl;
     return n1;}//如果a<b,则商为0，余数为被除数
     int t=La-Lb;//除被数和除数的位数之差
     for(int i=La-1;i>=0;i--)//将除数扩大10^t倍
        if(i>=t) b[i]=b[i-t];
        else b[i]=0;
     Lb=La;
     for(int j=0;j<=t;j++)
     {
         int temp;
         while((temp=sub(a,b+j,La,Lb-j))>=0)//如果被除数比除数大继续减
         {
             La=temp;
             r[t-j]++;
         }
     }
     for(i=0;i<L-10;i++) r[i+1]+=r[i]/10,r[i]%=10;//统一处理进位
     while(!r[i]) i--;//将整形数组表示的商转化成字符串表示的
     while(i>=0) s+=r[i--]+'0';
     //cout<<s<<endl;
     i=tp;
     while(!a[i]) i--;//将整形数组表示的余数转化成字符串表示的</span>
     while(i>=0) v+=a[i--]+'0';
     if(v.empty()) v="0";
     //cout<<v<<endl;
     if(nn==1) return s;//返回商
     if(nn==2) return v;//返回余数
}
string div(string a,int b)//高精度a除以单精度b
{
    string r,ans;
    int d=0;
    if(a=="0") return a;//特判
    for(int i=0;i<a.size();i++)
    {
            r+=(d*10+a[i]-'0')/b+'0';//求出商
            d=(d*10+(a[i]-'0'))%b;//求出余数
    }
    int p=0;
    for(int i=0;i<r.size();i++)
    if(r[i]!='0') {p=i;break;}
    return r.substr(p);
}
```

### 高精度幂

```c

//高精度幂(nlog(n)log(n))
#define L(x) (1 << (x))
const double PI = acos(-1.0);
const int Maxn = 133015;
double ax[Maxn], ay[Maxn], bx[Maxn], by[Maxn];
char sa[Maxn/2],sb[Maxn/2];
int sum[Maxn];
int x1[Maxn],x2[Maxn];
int revv(int x, int bits)
{
    int ret = 0;
    for (int i = 0; i < bits; i++)
    {
        ret <<= 1;
        ret |= x & 1;
        x >>= 1;
    }
    return ret;
}
void fft(double * a, double * b, int n, bool rev)
{
    int bits = 0;
    while (1 << bits < n) ++bits;
    for (int i = 0; i < n; i++)
    {
        int j = revv(i, bits);
        if (i < j)
            swap(a[i], a[j]), swap(b[i], b[j]);
    }
    for (int len = 2; len <= n; len <<= 1)
    {
        int half = len >> 1;
        double wmx = cos(2 * PI / len), wmy = sin(2 * PI / len);
        if (rev) wmy = -wmy;
        for (int i = 0; i < n; i += len)
        {
            double wx = 1, wy = 0;
            for (int j = 0; j < half; j++)
            {
                double cx = a[i + j], cy = b[i + j];
                double dx = a[i + j + half], dy = b[i + j + half];
                double ex = dx * wx - dy * wy, ey = dx * wy + dy * wx;
                a[i + j] = cx + ex, b[i + j] = cy + ey;
                a[i + j + half] = cx - ex, b[i + j + half] = cy - ey;
                double wnx = wx * wmx - wy * wmy, wny = wx * wmy + wy * wmx;
                wx = wnx, wy = wny;
            }
        }
    }
    if (rev)
    {
        for (int i = 0; i < n; i++)
            a[i] /= n, b[i] /= n;
    }
}
int solve(int a[],int na,int b[],int nb,int ans[])
{
    int len = max(na, nb), ln;
    for(ln=0; L(ln)<len; ++ln);
    len=L(++ln);
    for (int i = 0; i < len ; ++i)
    {
        if (i >= na) ax[i] = 0, ay[i] =0;
        else ax[i] = a[i], ay[i] = 0;
    }
    fft(ax, ay, len, 0);
    for (int i = 0; i < len; ++i)
    {
        if (i >= nb) bx[i] = 0, by[i] = 0;
        else bx[i] = b[i], by[i] = 0;
    }
    fft(bx, by, len, 0);
    for (int i = 0; i < len; ++i)
    {
        double cx = ax[i] * bx[i] - ay[i] * by[i];
        double cy = ax[i] * by[i] + ay[i] * bx[i];
        ax[i] = cx, ay[i] = cy;
    }
    fft(ax, ay, len, 1);
    for (int i = 0; i < len; ++i)
        ans[i] = (int)(ax[i] + 0.5);
    return len;
}
string mul(string sa,string sb)
{
    int l1,l2,l;
    int i;
    string ans;
    memset(sum, 0, sizeof(sum));
    l1 = sa.size();
    l2 = sb.size();
    for(i = 0; i < l1; i++)
        x1[i] = sa[l1 - i - 1]-'0';
    for(i = 0; i < l2; i++)
        x2[i] = sb[l2-i-1]-'0';
    l = solve(x1, l1, x2, l2, sum);
    for(i = 0; i<l || sum[i] >= 10; i++) // 进位
    {
        sum[i + 1] += sum[i] / 10;
        sum[i] %= 10;
    }
    l = i;
    while(sum[l] <= 0 && l>0)    l--; // 检索最高位
    for(i = l; i >= 0; i--)    ans+=sum[i] + '0'; // 倒序输出
    return ans;
}
string Pow(string a,int n)
{
    if(n==0) return 1;
    if(n==1) return a;
    if(n&1) return mul(Pow(a,n-1),a);
    string ans=Pow(a,n/2);
    return mul(ans,ans);
}
```

### 阶层

```c

//阶层
string fac(int n)
{
    const int L=100005;
    int a[L];
    string ans;
    if(n==0) return "1";
    fill(a,a+L,0);
    int s=0,m=n;
    while(m) a[++s]=m%10,m/=10;
    for(int i=n-1;i>=2;i--)
    {
        int w=0;
        for(int j=1;j<=s;j++) a[j]=a[j]*i+w,w=a[j]/10,a[j]=a[j]%10;
        while(w) a[++s]=w%10,w/=10;
    }
    while(!a[s]) s--;
    while(s>=1) ans+=a[s--]+'0';
    return ans;
}
```

### 求最大公约数

```c

//gcd
string add(string a,string b)
{
    const int L=1e5;
    string ans;
    int na[L]={0},nb[L]={0};
    int la=a.size(),lb=b.size();
    for(int i=0;i<la;i++) na[la-1-i]=a[i]-'0';
    for(int i=0;i<lb;i++) nb[lb-1-i]=b[i]-'0';
    int lmax=la>lb?la:lb;
    for(int i=0;i<lmax;i++) na[i]+=nb[i],na[i+1]+=na[i]/10,na[i]%=10;
    if(na[lmax]) lmax++;
    for(int i=lmax-1;i>=0;i--) ans+=na[i]+'0';
    return ans;
}
string mul(string a,string b)
{
    const int L=1e5;
    string s;
    int na[L],nb[L],nc[L],La=a.size(),Lb=b.size();//na存储被乘数，nb存储乘数，nc存储积
    fill(na,na+L,0);fill(nb,nb+L,0);fill(nc,nc+L,0);//将na,nb,nc都置为0
    for(int i=La-1;i>=0;i--) na[La-i]=a[i]-'0';//将字符串表示的大整形数转成i整形数组表示的大整形数
    for(int i=Lb-1;i>=0;i--) nb[Lb-i]=b[i]-'0';
    for(int i=1;i<=La;i++)
        for(int j=1;j<=Lb;j++)
        nc[i+j-1]+=na[i]*nb[j];//a的第i位乘以b的第j位为积的第i+j-1位（先不考虑进位）
    for(int i=1;i<=La+Lb;i++)
        nc[i+1]+=nc[i]/10,nc[i]%=10;//统一处理进位
    if(nc[La+Lb]) s+=nc[La+Lb]+'0';//判断第i+j位上的数字是不是0
    for(int i=La+Lb-1;i>=1;i--)
        s+=nc[i]+'0';//将整形数组转成字符串
    return s;
}
int sub(int *a,int *b,int La,int Lb)
{
    if(La<Lb) return -1;//如果a小于b，则返回-1
    if(La==Lb)
    {
        for(int i=La-1;i>=0;i--)
            if(a[i]>b[i]) break;
            else if(a[i]<b[i]) return -1;//如果a小于b，则返回-1

    }
    for(int i=0;i<La;i++)//高精度减法
    {
        a[i]-=b[i];
        if(a[i]<0) a[i]+=10,a[i+1]--;
    }
    for(int i=La-1;i>=0;i--)
        if(a[i]) return i+1;//返回差的位数
    return 0;//返回差的位数

}
string div(string n1,string n2,int nn)//n1,n2是字符串表示的被除数，除数,nn是选择返回商还是余数
{
    const int L=1e5;
    string s,v;//s存商,v存余数
     int a[L],b[L],r[L],La=n1.size(),Lb=n2.size(),i,tp=La;//a，b是整形数组表示被除数，除数，tp保存被除数的长度
     fill(a,a+L,0);fill(b,b+L,0);fill(r,r+L,0);//数组元素都置为0
     for(i=La-1;i>=0;i--) a[La-1-i]=n1[i]-'0';
     for(i=Lb-1;i>=0;i--) b[Lb-1-i]=n2[i]-'0';
     if(La<Lb || (La==Lb && n1<n2)) {
            //cout<<0<<endl;
     return n1;}//如果a<b,则商为0，余数为被除数
     int t=La-Lb;//除被数和除数的位数之差
     for(int i=La-1;i>=0;i--)//将除数扩大10^t倍
        if(i>=t) b[i]=b[i-t];
        else b[i]=0;
     Lb=La;
     for(int j=0;j<=t;j++)
     {
         int temp;
         while((temp=sub(a,b+j,La,Lb-j))>=0)//如果被除数比除数大继续减
         {
             La=temp;
             r[t-j]++;
         }
     }
     for(i=0;i<L-10;i++) r[i+1]+=r[i]/10,r[i]%=10;//统一处理进位
     while(!r[i]) i--;//将整形数组表示的商转化成字符串表示的
     while(i>=0) s+=r[i--]+'0';
     //cout<<s<<endl;
     i=tp;
     while(!a[i]) i--;//将整形数组表示的余数转化成字符串表示的</span>
     while(i>=0) v+=a[i--]+'0';
     if(v.empty()) v="0";
     //cout<<v<<endl;
     if(nn==1) return s;
     if(nn==2) return v;
}
bool judge(string s)//判断s是否为全0串
{
    for(int i=0;i<s.size();i++)
        if(s[i]!='0') return false;
    return true;
}
string gcd(string a,string b)//求最大公约数
{
    string t;
    while(!judge(b))//如果余数不为0，继续除
    {
        t=a;//保存被除数的值
        a=b;//用除数替换被除数
        b=div(t,b,2);//用余数替换除数
    }
    return a;
}
```

### 取模（高精）

```c

//取模
int mod(string a,int b)//高精度a除以单精度b
{
    int d=0;
    for(int i=0;i<a.size();i++) d=(d*10+(a[i]-'0'))%b;//求出余数
    return d;
}  
```

### 进制转换（将字符串表示的10进制大整数转换为m进制的大整数）

```c

//进制转换
//将字符串表示的10进制大整数转换为m进制的大整数
//并返回m进制大整数的字符串
bool judge(string s)//判断串是否为全零串
{
    for(int i=0;i<s.size();i++)
        if(s[i]!='0') return 1;
    return 0;
}
string solve(string s,int n,int m)//n进制转m进制只限0-9进制，若涉及带字母的进制，稍作修改即可
{
    string r,ans;
    int d=0;
    if(!judge(s)) return "0";//特判
    while(judge(s))//被除数不为0则继续
    {
        for(int i=0;i<s.size();i++)
        {
            r+=(d*n+s[i]-'0')/m+'0';//求出商
            d=(d*n+(s[i]-'0'))%m;//求出余数
        }
       s=r;//把商赋给下一次的被除数
       r="";//把商清空
        ans+=d+'0';//加上进制转换后数字
        d=0;//清空余数
    }
    reverse(ans.begin(),ans.end());//倒置下
    return ans;
}
```

## 计算几何

### 基础模板

```c
//点和线的表示
typedef long double LD;
struct PII{
	int x,y;
	bool operator<(const Point &o)const{
		if(x==o.x) return y<o.y;
		return x<o.x;
	}
};
struct Point{
	double x,y;
	bool operator<(const Point &o)const{
		if(x==o.x) return y<o.y;
		return x<o.x;
	}
};
struct Line{
	Point st,ed;
};
//判正负
int sign(double a){
	if(fabs(a)<=eps) return 0;
	return a>0?1:-1;
}
//比较大小
int dcmp(double a,double b){
	if(fabs(a-b)<eps) return 0;
	return a>b?1:-1;
}
```

### 点积

```c
double Dot(Point a, Point b) {
	return a.x*b.x+a.y*b.y;
}
```

### 叉积

```c
double cross(Point a,Point b){ 
	return a.x*b.y-a.y*b.x;
}
```

### 模长

```c
double ABS(Point a){
	return sqrt(a.x*a.x+a.y*a.y);
}
double norm(Point a){
    return a.x*a.x+a.y*a.y;
}
```

### 两个向量是否同象限

```c
bool same_quadrant(Point v,Point p) {
	LD a=v.x, b=v.y, c=p.x, d=p.y;
	int aa=sign(a), bb=sign(b);
	int cc=sign(c), dd=sign(d);
	return aa*cc>=0 && bb*dd>=0;
}
```

### 两向量是否共线

```c
int dcmp(double x) {
	if (fabs(x)<eps) return 0;
	else if (x<0) return -1;
	else return 1;
}
double cross(Point a,Point b){ 
	return a.x*b.y-a.y*b.x;
}
bool on_line(Point a, Point b) {	//a和b是否共线 
	return dcmp(cross(a,b))==0;
}
```

### 两向量是否垂直

```c
double Dot(Point a, Point b) {
	return a.x*b.x+a.y*b.y;
}
bool is_vertical(Point a, Point b) {
	return dcmp(Dot(a,b))==0;
}
```

### 两个向量是否同方向

```c
int same_direction(Point v, Point p) {    //判断向量v和向量p是否共线且同向
	if (on_line(v, p) && same_quadrant(v,p)) return 1;    //同向
	else if(on_line(v, p) && !same_quadrant(v,p)) return -1;    //反向
	return 0;    //不共线
}
```

### 点旋转

```c
//点逆时针旋转a度后的坐标
Point rotate1(Point p,double a){
	return {p.x*cos(a)-p.y*sin(a),p.x*sin(a)+p.y*cos(a)};
}
//点顺时针旋转a度后的坐标
Point rotate2(Point p,double a){
	return {p.x*cos(a)+p.y*sin(a),-p.x*sin(a)+p.y*cos(a)};
}
```

### 两点之间的距离

```c
double getdis(Point a,Point b) {
	return hypot(a.x-b.x,a.y-b.y);
}
```

### 直线

#### 直线和线段是否相交

```c
//直线a是否经过线段b 
bool on_segment(Line a,Line b){
	point q1=a.st,q2=a.ed;
	if(sign(area(q1,q2,b.st))*sign(area(q1,q2,b.ed))>0) return 0;
	return 1;
}
```

#### 直线和直线的交点

```c
//求两直线的交点
Point get_line_intersection(Point p, Point v, Point q, Point w){
    Point u = p - q;
    LD t = cross(w, u) / cross(v, w);
    return {p.x + t * v.x, p.y + t * v.y};
}
Point get_line_intersection(Line& a, Line& b){
    return get_line_intersection(a.st, a.ed - a.st, b.st, b.ed - b.st);
}
```

#### 判断b和c的交点是否在直线右面

```c
bool on_right(Line& a, Line& b, Line& c){
    auto o = get_line_intersection(b, c);
    return sign(area(a.st, a.ed, o)) < 0;
}
```

#### 得到直线的角度

```c
double get_angle(const Line &a){
	return atan2(a.ed.y-a.st.y,a.ed.x-a.st.x);
}
```

#### 求点关于直线的对称点

```c
Point point_line(Line l, Point p) { //点p关于直线l的对称点
	Point p1 = l.st;
	Point p2 = l.ed;
	double _x, _y;
	if(p1.x - p2.x == 0) { //l斜率不存在
		_x = 2 * p1.x - p.x;
		_y = p.y;
		return Point{_x,_y};
	} else if(p1.y - p2.y == 0) { //l斜率为0
		_x = p.x;
		_y = 2 * p1.y - p.y;
		return Point{_x,_y};
	} else {
		double k1 = (p1.y - p2.y) / (p1.x - p2.x);
		double b1 = p1.y - k1 * p1.x;
		double k2 = -1 / k1;
		double b2 = p.y - k2 * p.x;
		_x = (b2 - b1) / (k1 - k2);
		_y = k2 * _x + b2;
		return Point{2 * _x - p.x, 2 * _y - p.y};
	}
}
```

#### 点到直线的距离

```c
double PLDis(Point a,Line s) {	//点到直线的距离
    double A=s.st.y-s.ed.y;
    double B=s.ed.x-s.st.x;
    double C=(s.st.x-s.ed.x)*s.st.y-(s.st.y-s.ed.y)*s.st.x;
    return fabs(A*a.x+B*a.y+C)/sqrt((A*A+B*B));
}
```

#### 点在直线的投影

```c
//点p在直线s上的投影
Point Projection(Point p,Line s) {
    Point alp=p-s.st;
    Point beta=s.ed-s.st;
    double res=Dot(alp,beta)/norm(beta);	//norm(): 模长的平方
    return s.st+(res*beta);
}
```

#### 求多边形的面积

```c
vector<Point> polygon;
double get_Area(vector<Point> polygon) {
	double ans=0;
	int n=(int)polygon.size();
	for(int i=0; i<n; i++) {
		ans+=cross(polygon[i],polygon[(i+1)%n]);
	}
	return fabs(ans/2);
}
```

### 线段

#### 判断一个点是否在一个线段上

```c
bool on_segment(Point p, Point a, Point b){
	return sign(cross(p - a, p - b)) == 0 && sign(dot(p - a, p - b)) <= 0;
}
```

#### 判断两个线段是否相交

```c
bool segment_intersection(Point a1, Point a2, Point b1, Point b2){
	double c1 = cross(a2 - a1, b1 - a1), c2 = cross(a2 - a1, b2 - a1);
	double c3 = cross(b2 - b1, a2 - b1), c4 = cross(b2 - b1, a1 - b1);
	return sign(c1) * sign(c2) <= 0 && sign(c3) * sign(c4) <= 0;
}
```

#### 判断线和线段是否相交

```c
bool line_segment_intersection(Line a,Line b){
	if(on_line(a.p2-a.p1,b.p2-b.p1)){
		if(on_line(a.p2-a.p1,b.p1-a.p1)) return 1;
		else return 0;
	} 
	Point o=get_line_intersection(a,b);
	if(on_segment(o,b)) return 1;
	else return 0;
}
```

### 圆

#### 直线和圆的交点

```c
int CCL(Line s,Point o,double r,Point &o1,Point &o2) {
    Point x=Projection(o,s);
    double dis=PLDis(o,s);
    if(dis>r) { //距离>r没有交点 
        return 0;
    }
    if(dis==r) { //只有一个交点
        o1=x;
        return 1;
    }
    double beta=sqrt(r*r-dis*dis);//勾股定理
    Point pp=s.ed-s.st;
    pp=pp/pp.ABS();//单位向量
    Point ans1=x-beta*pp;
    Point ans2=x+beta*pp;
    o1=ans1;
    o2=ans2;
    return 2;
}
```

### 凸包

```c
//极角排序比较函数
bool cmp(PDD a,PDD b){
    a=a-bas; b=b-bas;
	double ag1=atan2(a.y,a.x),ag2=atan2(b.y,b.x);
    if(ag1==ag2) a.x<b.x;
	else return ag1<ag2;
}
//二维凸包
void get_convex(){
	sort(q+1,q+1+n,cmp);
	stk[++top]=q[1];
	stk[++top]=q[2];
	for(int i=3;i<=n;i++){
		while(top>=2 && area(stk[top-1],stk[top],q[i])<=0) --top;
		stk[++top]=q[i];
	}
	return ;
}
```

### 三维凸包

```c
#include <bits/stdc++.h>
#define ios ios::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define endl '\n'
using namespace std;
typedef long double LD;
const LD eps=1e-10;
const int N=110;
const double PI=acos(-1);
double rand_eps(){
	return ((double)rand()/RAND_MAX-0.5)*eps;
}
struct Point{
	double x,y,z;
	void shake(){	//微小扰动
		x+=rand_eps();
		y+=rand_eps();
		z+=rand_eps();
	}
	Point operator - (Point a){
		return {x-a.x,y-a.y,z-a.z};
	}
	Point operator + (Point a){
		return {x+a.x,y+a.y,z+a.z};
	}
	Point operator * (Point a){	//向量叉乘
		return {y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x};
	}
	double operator & (Point a){	//向量点积
		return x*a.x+y*a.y+z*a.z;
	}
	double len(){	//向量模长
		return sqrt(x*x+y*y+z*z);
	}
}q[N];
struct Plane{
	int v[3];
	Point norm(){
		return (q[v[1]]-q[v[0]])*(q[v[2]]-q[v[0]]);	//返回法向量
	}
	double area(){
		return norm().len()/2;	//面积
	}
	bool above(Point a){
		return ((a-q[v[0]]) & norm()) >= 0;	//返回一个点是否在一个平面上方，也就是平面能不能被照到
	}
}pl[N],bk[N];
bool g[N][N];
int n,tot;
void get_convex(){
	pl[++tot]={1,2,3};	//放进去前3个点组成的两个平面（正反）
	pl[++tot]={1,3,2};
	for(int i=4;i<=n;i++){
		int cnt=0;	//把更新后的平面放进去备份数组
		for(int j=1;j<=tot;j++){
			bool flag=pl[j].above(q[i]);	//表示q[i]照到了第j个平面
			if(!flag) bk[++cnt]=pl[j];	//没有照到
			for(int k=0;k<3;k++){
				g[pl[j].v[k]][pl[j].v[(k+1)%3]]=flag;	//标记每一条边是否照到
			}
		}
		for(int j=1;j<=tot;j++){
			for(int k=0;k<3;k++){
				int a=pl[j].v[k],b=pl[j].v[(k+1)%3];	//两点的编号
				if(g[a][b] && !g[b][a]) bk[++cnt]={a,b,i};	//正着可以照到，反着照不到，把照到的边扔掉
			}
		}
		tot=cnt;	//更新面的数量
		for(int i=1;i<=tot;i++) pl[i]=bk[i];	//更新所有的面
	}
}
int main()
{
	ios;
	cin>>n;
	for(int i=1;i<=n;i++){
		cin>>q[i].x>>q[i].y>>q[i].z;
		q[i].shake();
	}
	get_convex();
	double ans=0;
	for(int i=1;i<=tot;i++){
		ans+=pl[i].area();
	}
	cout<<fixed<<setprecision(6)<<ans<<endl;
	
	return 0;
 } 
```

### 半平面交

```c
//极角排序比较函数
bool cmp(PDD a,PDD b){
    a=a-bas; b=b-bas;
	double ag1=atan2(a.y,a.x),ag2=atan2(b.y,b.x);
    if(ag1==ag2) a.x<b.x;
	else return ag1<ag2;
}
//二维凸包
void get_convex(){
	sort(q+1,q+1+n,cmp);
	stk[++top]=q[1];
	stk[++top]=q[2];
	for(int i=3;i<=n;i++){
		while(top>=2 && area(stk[top-1],stk[top],q[i])<=0) --top;
		stk[++top]=q[i];
	}
	return ;
}
```

### 最小圆覆盖

```c
#include <bits/stdc++.h>
#define ios ios::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define endl '\n'
#define x first
#define y second
using namespace std;
typedef long double LD;
const LD eps=1e-12;
const int N=101000;
const double PI=acos(-1);
//点和线的表示
typedef long double LD;
typedef pair<double,double> PDD;
struct Circle {
	PDD p;

	double r;
};
//重载运算符"-"
PDD operator-(const PDD &a,const PDD &b) {
	return {a.x-b.x,a.y-b.y};
}
PDD operator+(const PDD &a,const PDD &b) {
	return {a.x+b.x,a.y+b.y};
}
PDD operator/ (const PDD &a,double t) {
	return {a.x/t,a.y/t};
}
//求向量a和b的叉积
double cross(PDD a,PDD b) {
	return a.x*b.y-a.y*b.x;
}
//求向量ab和向量ac的叉积，也就是abc三角形的面积，顺时针为负，逆时针为正
double area(PDD a,PDD b,PDD c) {
	return cross(b-a,c-a);
}
//判正负
int sign(double a) {
	if(fabs(a)<=eps) return 0;
	return a>0?1:-1;
}
//比较大小
int fcmp(double a,double b) {
	if(fabs(a-b)<eps) return 0;
	return a>b?1:-1;
}
PDD get_line_intersection(PDD p, PDD v, PDD q, PDD w) {
	PDD u = p - q;
	LD t = cross(w, u) / cross(v, w);
	return {p.x + t * v.x, p.y + t * v.y};
}
//点逆时针旋转a度后的坐标
PDD rotate(PDD p,double a) {
	return {p.x*cos(a)-p.y*sin(a),p.x*sin(a)+p.y*cos(a)};
}
pair<PDD,PDD> getline(PDD a,PDD b) {
	return {(a+b)/2,rotate(b-a,PI/2)};
}
double getdis(PDD a,PDD b) {
	return sqrt(pow(b.y-a.y,2)+pow(b.x-a.x,2));
}
Circle get_circle(PDD a,PDD b,PDD c) {
	auto u=getline(a,b),v=getline(a,c);
	auto p=get_line_intersection(u.x,u.y,v.x,v.y);
	return {p,getdis(p,a)};
}
PDD q[N];
int n;
Circle get_Circle() {
	random_shuffle(q+1,q+1+n);
	Circle c;
	c.p=q[1];
	c.r=0;
	for(int i=2; i<=n; i++) {
		if(fcmp(c.r,getdis(c.p,q[i]))<0) {
			c= {q[i],0};
			for(int j=1; j<i; j++) {
				if(fcmp(c.r,getdis(c.p,q[j]))<0) {
					c= {(q[i]+q[j])/2,getdis(q[i],q[j])/2};
					for(int k=1; k<j; k++) {
						if(fcmp(c.r,getdis(c.p,q[k]))<0) {
							c=get_circle(q[i],q[j],q[k]);
						}
					}
				}
			}
		}
	}
	return c;
}

int main() {
	ios;
	cin>>n;
	for(int i=1; i<=n; i++) cin>>q[i].x>>q[i].y;
	Circle c;
	c=get_Circle();
	cout<<fixed<<setprecision(10)<<c.r<<endl;
	cout<<fixed<<setprecision(10)<<c.p.x<<" "<<c.p.y<<endl;
	return 0;
}
```

