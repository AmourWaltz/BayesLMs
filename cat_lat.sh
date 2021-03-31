var1=3
var2=$[3 + 3]
var3=$[$var1 + $var2]
var4=$[$var3 * 2]
var5=$[2 * ($var3 - $var4)]
echo "var1=$var1,var2=$var2,var3=$var3,var4=$var4,var5=$var5"

var6=$[100 / 30]
echo "var6=$var6"

for i in {1..5}; do
	j=$[$i * 6]
	cat lat.$[$j-5].gz lat.$[$j-4].gz lat.$[$j-3].gz lat.$[$j-2].gz lat.$[$j-1].gz lat.$[$j-0].gz > lat.$i.new.gz
	echo "i=$i,j=$j"
done
