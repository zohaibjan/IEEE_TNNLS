function [x,y]=balance_data(x,y)
t=unique(y);
r=individual_count(y);
max_r=max(r);
for i=1:length(t)
    id=y==t(i);
    a=x(id,:);
    [mu,std]=compute_center_std(a);
    m=(max_r-r(i));
    for j=1:m
        x=[x;normrnd(mu,std.^2,[1,size(x,2)])];
        y=[y;t(i)];
    end
end

end

function [mu,std]=compute_center_std(a)
n=size(a,1);

for i=1:size(a,2)
    mu(i)=sum(a(:,i))/n;
    b=(mu(i)-a(:,i));
    std(i)=max(abs(mu(i)-a(:,i)));
end

end

function [count]=individual_count(y)
a=unique(y);
count=[];
for i=1:length(a)
   count(i)=sum(y==a(i));
end
end