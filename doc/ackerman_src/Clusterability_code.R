
if(!require(devtools)) install.packages("devtools",repos="http://cran.us.r-project.org")
devtools::install_github("kassambara/factoextra")
install.packages("clustertend",repos="http://cran.us.r-project.org")
install.packages("seriation",repos="http://cran.us.r-project.org")
install.packages("diptest",repos="http://cran.us.r-project.org")
install.packages("silvermantest",repos="http://cran.us.r-project.org")
install.packages("lubridate",repos="http://cran.us.r-project.org")
install.packages("princurve",repos="http://cran.us.r-project.org")
library(princurve)
library(lubridate)
library(factoextra)
library(clustertend)
library(seriation)
library("diptest")
library("silvermantest")

Tclust<-function(n, xMean, yMean, df){
	x<-matrix(rt(n, df, xMean), nrow=n, ncol = 1)
	y<-matrix(rt(n, df, yMean), nrow=n, ncol = 1)
	t<-cbind(x,y)
	return(t)
}

twoTclust<-function(n, df){
	t1<-Tclust(n,50,50,df)
	t2<-Tclust(n,50,50,df)+100
	t<-rbind(t1,t2)
	return(t)
}

cluster<-function(n,xMean,yMean,sd){
	x<-matrix(rnorm(n,xMean,sd),nrow=n, ncol = 1)
	y<-matrix(rnorm(n,yMean,sd),nrow=n, ncol = 1)
	c<-cbind(x,y)
	return(c)
}

naomicustomcluster<-function(n, sd){
	x<-matrix(rnorm(n,10,sd),nrow=n, ncol = 1)
	y<-matrix(rnorm(n,100,sd),nrow=n, ncol = 1)
	c<-cbind(x,y)
	return(c)
}

parallelLines<-function(n,spread,sd){
	l1<-matrix(30,nrow=n, ncol = 1)
	l2<-matrix(rnorm(n,50,sd),nrow=n, ncol = 1) #runif(n,20,20+length)
	l<-cbind(l1,l2)
	l3<-matrix(30+spread,nrow=n, ncol = 1)
	l4<-matrix(rnorm(n,50,sd),nrow=n, ncol = 1) #runif(n,20,20+length)
	ll<-cbind(l3,l4)
	L<-rbind(l,ll)
	return(L)
}

Line<-function(n, spread, sd){
	l1<-matrix(30,nrow=n, ncol = 1)
	l2<-matrix(rnorm(n,50,sd),nrow=n, ncol = 1) #runif(n,20,20+length)
	l<-cbind(l1,l2)
	return(l)
}

Circle<-function(n){
	x1<-matrix(rnorm(n),nc=2)
	y1<-x1/sqrt(rowSums(x1^2))
	return(y1)
}

concentricCircles<-function(n){
	x1<-matrix(rnorm(n),nc=2)
	x2<-matrix(rnorm(n),nc=2)
	y1<-x1/sqrt(rowSums(x1^2))
	y2<-x2/sqrt(rowSums(x2^2))
	y2<-y2*2
	c<-rbind(y1,y2)
	return(c)
}

threeConcentricCircles<-function(n){
	x1<-matrix(rnorm(n),nc=2)
	x2<-matrix(rnorm(n),nc=2)
	x3<-matrix(rnorm(n),nc=2)
	c1<-x1/sqrt(rowSums(x1^2))
	c2<-x2/sqrt(rowSums(x2^2))
	c3<-x3/sqrt(rowSums(x3^2))
	c2<-c2*2
	c3<-c3*3
	c<-rbind(c1,c2,c3)
	return(c)
}

fiveConcentricCircles<-function(n){
	x1<-matrix(rnorm(n),nc=2)
	x2<-matrix(rnorm(n),nc=2)
	x3<-matrix(rnorm(n),nc=2)
	x4<-matrix(rnorm(n),nc=2)
	x5<-matrix(rnorm(n),nc=2)
	c1<-x1/sqrt(rowSums(x1^2))
	c2<-x2/sqrt(rowSums(x2^2))
	c3<-x3/sqrt(rowSums(x3^2))
	c4<-x4/sqrt(rowSums(x4^2))
	c5<-x5/sqrt(rowSums(x5^2))
	c2<-c2*2
	c3<-c3*3
	c4<-c4*2
	c5<-c5*3
	c<-rbind(c1,c2,c3,c4,c5)
	return(c)
}

cluster3D<-function(n,xMean,yMean,zMean,sd){
	x<-matrix(rnorm(n,xMean,sd),nrow=n, ncol = 1)
	y<-matrix(rnorm(n,yMean,sd),nrow=n, ncol = 1)
	z<-matrix(rnorm(n,zMean,sd),nrow=n, ncol = 1)
	c<-cbind(x,y,z)
	return(c)
}

cluster10D<-function(n, x,sd){
	d1<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d2<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d3<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d4<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d5<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d6<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d7<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d8<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d9<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	d10<-matrix(rnorm(n,x*20,sd),nrow=n, ncol = 1)
	c<-cbind(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10)
	return(c)
}

closeCluster10D<-function(n, x,sd){
	d1<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d2<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d3<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d4<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d5<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d6<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d7<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d8<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d9<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	d10<-matrix(rnorm(n,x*10,sd),nrow=n, ncol = 1)
	c<-cbind(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10)
	return(c)
}

cluster50D<-function(n, scale,sd){
	d1<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d2<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d3<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d4<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d5<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d6<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d7<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d8<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d9<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d10<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d11<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d12<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d13<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d14<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d15<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d16<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d17<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d18<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d19<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d20<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d21<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d22<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d23<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d24<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d25<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d26<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d27<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d28<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d29<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d30<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d31<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d32<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d33<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d34<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d35<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d36<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d37<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d38<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d39<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d40<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d41<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d42<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d43<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d44<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d45<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d46<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d47<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d48<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d49<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	d50<-matrix(rnorm(n,scale,sd),nrow=n, ncol = 1)
	c<-cbind(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,
			d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31,
			d32,d33,d34,d35,d36,d37,d38,d39,d40,d41,d42,d43,d44,d45,
			d46,d47,d48,d49,d50)
	return(c)
}


noise<-function(n,xMean,yMean){
	x<-matrix(rnorm(n,xMean,20),nrow=n, ncol = 1)
	y<-matrix(rnorm(n,yMean,20),nrow=n, ncol = 1)
	c<-cbind(x,y)
	return(c)
}

clusterWithOutlier<-function(){
	c<-cluster(50, 50, 50,2)
	o<-cluster(1, sample(60:65, 1),sample(60:65 ,1),2)
	g<-rbind(c,o)
	return(g)
}

largeClusterWithOutlier<-function(){
	c<-cluster(250,50,50,2)
	o<-cluster(1, sample(60:65, 1), sample(60:65, 1), 2)
	g<-rbind(c,o)
	return(g)
}

clusterWith3Outliers<-function(){
	c<-cluster(50,50,50,2)
	o1<-cluster(1,sample(40:45, 1),sample(55:60, 1),2)
	o2<-cluster(1,sample(65:70, 1),sample(65:70, 1),2)
	o3<-cluster(1,sample(65:70, 1),sample(45:50, 1),2)
	g<-rbind(c,o1,o2,o3)
	return(g)
}

twoSeparatedClusters<-function(){
	c1<-cluster(50,30,30,2)
	c2<-cluster(50,50,50,2)
	g<-rbind(c1,c2)
	return(g)
}

threeSeparatedClusters<-function(){
	c1<-cluster(50,35,40,2)
	c2<-cluster(50,65,40,2)
	c3<-cluster(50,50,60,2)
	g<-rbind(c1,c2,c3)
	return(g)
}

threeSeparatedClusters3D<-function(){
	c1<-cluster3D(50,20,20,20,2)
	c2<-cluster3D(50,40,40,40,2)
	c3<-cluster3D(50,60,60,60,2)
	g<-rbind(c1,c2,c3)
	return(g)
}

threeClustersVaryingDensity<-function(){
	c1<-cluster(100,35,40,2)
	c2<-cluster(66,65,40,2)
	c3<-cluster(33,50,60,2)
	g<-rbind(c1,c2,c3)
	return(g)	
}

threeCloseClusters<-function(){
	c1<-cluster(50,30,20,2)
	c2<-cluster(50,40,20,2)
	c3<-cluster(50,35,30,2)
	g<-rbind(c1,c2,c3)
	return(g)
}

threeNoisyClusters<-function(){
	c1<-cluster(50,30,40,2)
	c2<-cluster(50,70,40,2)
	c3<-cluster(50,50,80,2)
	n<-noise(80,50,50)
	g<-rbind(c1,c2,c3,n)
	return(g)
}

threeClustersVaryingDiameter<-function(){
	c1<-cluster(50,30,40,1)
	c2<-cluster(50,70,40,3)
	c3<-cluster(50,50,80,5)
	g<-rbind(c1,c2,c3)
	return(g)
}

twoSeparatedClusters10D<-function(){
	c1<-cluster10D(50,1,2)
	c2<-cluster10D(50,2,2)
	g<-rbind(c1,c2)
	return(g)
}

twoCloseClusters10D<-function(){
	c1<-closeCluster10D(50,1,2)
	c2<-closeCluster10D(50,2,2)
	g<-rbind(c1,c2)
	return(g)
}

twoClusters50D<-function(scale){
	c1<-cluster50D(100,1*scale,2)
	c2<-cluster50D(100,2*scale,2)
	g<-rbind(c1,c2)
	return(g)
}

fourSeparatedClusters10D<-function(){
	c1<-cluster10D(50,1,2)
	c2<-cluster10D(50,2,2)
	c3<-cluster10D(50,3,2)
	c4<-cluster10D(50,4,2)
	g<-rbind(c1,c2,c3,c4)
	return(g)
}

tenSeparatedClusters<-function(){
	c1<-cluster(30,35,40,2)
	c2<-cluster(30,65,40,2)
	c3<-cluster(30,50,60,2)
	c4<-cluster(30,50,15,2)
	c5<-cluster(30,80,20,2)
	c6<-cluster(30,10,60,2)
	c7<-cluster(30,15,25,2)
	c8<-cluster(30,35,80,2)
	c9<-cluster(30,85,90,2)
	c10<-cluster(30,95,50,2)
	g<-rbind(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10)
	return(g)
}

tenCloseClusters<-function(){
	c1<-cluster(30,30,20,2)
	c2<-cluster(30,40,20,2)
	c3<-cluster(30,35,30,2)
	c4<-cluster(30,25,10,2)
	c5<-cluster(30,25,30,2)
	c6<-cluster(30,35,10,2)
	c7<-cluster(30,50,30,2)
	c8<-cluster(30,30,40,2)
	c9<-cluster(30,20,20,2)
	c10<-cluster(30,45,40,2)
	g<-rbind(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10)
	return(g)
}

sevenCloseDenseClusters<-function(){
	c1<-cluster(50,30,20,2)
	c2<-cluster(50,40,20,2)
	c3<-cluster(50,35,30,2)
	c4<-cluster(50,25,10,2)
	c5<-cluster(50,25,30,2)
	c6<-cluster(50,35,10,2)
	c7<-cluster(50,20,20,2)
	g<-rbind(c1,c2,c3,c4,c5,c6,c7)
	return(g)
}

sevenCloseSparseClusters<-function(){
	c1<-cluster(10,30,20,2)
	c2<-cluster(10,40,20,2)
	c3<-cluster(10,35,30,2)
	c4<-cluster(10,25,10,2)
	c5<-cluster(10,25,30,2)
	c6<-cluster(10,35,10,2)
	c7<-cluster(10,20,20,2)
	g<-rbind(c1,c2,c3,c4,c5,c6,c7)
	return(g)
}

fiveCloseClusters<-function(){
	c1<-cluster(30,30,20,2)
	c2<-cluster(30,35,30,2)
	c3<-cluster(30,25,10,2)
	c4<-cluster(30,25,30,2)
	c5<-cluster(30,35,10,2)
	g<-rbind(c1,c2,c3,c4,c5)
	return(g)
}

sevenCloseClusters<-function(){
	c1<-cluster(30,30,20,2)
	c2<-cluster(30,40,20,2)
	c3<-cluster(30,35,30,2)
	c4<-cluster(30,25,10,2)
	c5<-cluster(30,25,30,2)
	c6<-cluster(30,35,10,2)
	c7<-cluster(30,20,20,2)
	g<-rbind(c1,c2,c3,c4,c5,c6,c7)
	return(g)
}

irisDataset<-function(){
	g<-iris[,1:4]
	return(g)
}

digitsSubset<-function(a, b){
	#Digits<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\6.3\\dig.txt", header=FALSE, sep=",")
	Digits<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\8.15\\newDigits.txt", header=FALSE, sep=",")
	Digits<-Digits[Digits$V65== a | Digits$V65== b , ]
	Digits$V65 = NULL
	return(Digits)
}


digitsDataset<-function(){
	#Digits<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\6.3\\dig.txt", header=FALSE, sep=",")
	Digits<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\8.15\\newDigits.txt", header=FALSE, sep=",")
	Digits$V65 = NULL
	subset<-Digits[sample(1:nrow(Digits), (nrow(Digits)*0.2)),]
	return(subset)
}

digitsSet<-function(a){
	#Digits<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\6.3\\dig.txt", header=FALSE, sep=",")
	Digits<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\8.15\\newDigits.txt", header=FALSE, sep=",")
	Digits<-Digits[Digits$V65==a, ]
	Digits$V65 = NULL
	return(Digits)
}
cancerDataset<-function(){
	c<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\6.3\\cancer.txt", header=FALSE, sep=",")
	c$V2 = NULL
	c$V1 = NULL
	subset<-c[sample(1:nrow(c), (nrow(c)*0.6)),]
	return(subset)
}

musicDataset<-function(){
	mus<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\6.3\\music.txt", header=FALSE, sep=",")
	mus$V69 = NULL
	mus$V70 = NULL
	subset<-mus[sample(1:nrow(mus), (nrow(mus)*0.3)),]
	return(subset)
}

randuDataset<-function(){
	mus <- randu
	subset<-mus[sample(1:nrow(mus), (nrow(mus)*0.9)),]
	return(subset)
}

printTime<-function(prefix="", sec, suffix=""){
	cat(prefix)
	if(seconds_to_period(sec)$day > 0){	
		cat(seconds_to_period(sec)$day, "Days ") 
	}
	if(seconds_to_period(sec)$hour > 0){
		cat(seconds_to_period(sec)$hour, "Hours ")
	}
	if(seconds_to_period(sec)$minute > 0){
		cat(seconds_to_period(sec)$minute, "Minutes ")
	}
	cat(seconds_to_period(sec)$second, "Seconds\n", suffix)
	
}

digitsTrainingSet<-function(hop = 10){
	trainDigits<-read.table("C:\\Users\\zstur_000\\Desktop\\clustering\\8.15\\newDigits.txt", header=FALSE, sep=",")
	trainDigits$V65=NULL


	digDist<-dist(trainDigits)
	digDip<-dip.test(digDist)$p.value
	digSilv<-silverman.test(digDist,1,adjust=TRUE)@p_value

	digClassicDip<-dip.test(as.matrix(trainDigits))$p.value
	digClassicSilv<-silverman.test(as.matrix(trainDigits),1,adjust=TRUE)@p_value
	digPCADip<-dip.test(as.matrix(prcomp(trainDigits)$x[,1]))$p.value
	digPCASilv<-silverman.test(as.matrix(prcomp(trainDigits)$x[,1]),1,adjust=TRUE)@p_value

	digHop=rep(NA,hop)
	for(i in 1:hop){
		digHop[i]<-hopkins(trainDigits, n = nrow(trainDigits)/3)$H
	}

	result<-matrix(,1,7)
	colnames(result)<-c("Dip","Silverman","Hopkins","Hop<0.25","Hop>0.75","Classic Dip","Classic Silv.","PCA Dip","PCA Silv.")
	rownames(result)<-c("TrainDigits")
	result[1,1]<-round(digDip,4)
	result[1,2]<-round(digSilv,4)
	result[1,3]<-round(mean(digHop),4)
	result[1,4]<-round(sum(digHop<0.25),4)
	result[1,5]<-round(sum(digHop>0.75),4)
	result[1,6]<-round(digClassicDip,4)
	result[1,7]<-round(digClassicSilv,4)
	result[1,8]<-round(digPCADip,4)
	result[1,9]<-round(digPCASilv,4)

	return(result)
}

realData<-function(it){

	faithfulHop=rep(NA,it)
	carHop=rep(NA,it)
	beaverHop=rep(NA,it)
	usArrestHop=rep(NA,it)
	attitudeHop=rep(NA,it)
	irisHop=rep(NA,it)
	swissHop=rep(NA,it)
	treeHop=rep(NA,it)
	usJudgeRatingHop=rep(NA,it)
	riverHop=rep(NA,it)
	mtcarsHop=rep(NA,it)
	rockHop=rep(NA,it)

	checkpoint<-Sys.time()
	faithfulData<-dist(faithful)
	faithfulDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsData<-dist(cars)
	carsDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverData<-dist(beaver1)	
	beaverDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	usArrests<-dist(USArrests)
	usArrestsClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudeData<-dist(attitude)	
	attitudeDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisData<-dist(irisDataset())
	irisDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissData<-dist(swiss)
	swissDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treeData<-dist(trees)
	treeDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	usJudgeRatings<-dist(USJudgeRatings)
	usJudgeRatingsClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverData<-dist(rivers)
	riverDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsData<-dist(mtcars)
	mtcarsDataClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	rockData<-dist(rock)
	rockDataClock = as.numeric(Sys.time()-checkpoint)


	checkpoint<-Sys.time()
	faithfulSilv<-silverman.test(faithfulData,1,adjust=TRUE)@p_value
	faithfulSilvClock = faithfulDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsSilv<-silverman.test(carsData,1,adjust=TRUE)@p_value
	carsSilvClock = carsDataClock + as.numeric(Sys.time()-checkpoint)


	checkpoint<-Sys.time()
	beaverSilv<-silverman.test(beaverData,1,adjust=TRUE)@p_value
	beaverSilvClock = beaverDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestSilv<-silverman.test(usArrests,1,adjust=TRUE)@p_value
	USArrestSilvClock = usArrestsClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudeSilv<-silverman.test(attitudeData,1,adjust=TRUE)@p_value
	attitudeSilvClock = attitudeDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisSilv<-silverman.test(irisData,1,adjust=TRUE)@p_value
	irisSilvClock = irisDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissSilv<-silverman.test(swissData,1,adjust=TRUE)@p_value
	swissSilvClock = swissDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treeSilv<-silverman.test(treeData,1,adjust=TRUE)@p_value
	treeSilvClock = treeDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingSilv<-silverman.test(usJudgeRatings,1,adjust=TRUE)@p_value
	USJudgeRatingSilvClock = usJudgeRatingsClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverSilv<-silverman.test(riverData,1,adjust=TRUE)@p_value
	riverSilvClock = riverDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsSilv<-silverman.test(mtcarsData,1,adjust=TRUE)@p_value
	mtcarsSilvClock = mtcarsDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	rockSilv<-silverman.test(rockData,1,adjust=TRUE)@p_value
	rockSilvClock = rockDataClock + as.numeric(Sys.time()-checkpoint)


	checkpoint<-Sys.time()
	rockDip<-dip.test(rockData)$p.value
	rockDipClock = rockDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsDip<-dip.test(mtcarsData)$p.value
	mtcarsDipClock = mtcarsDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	faithfulDip<-dip.test(faithfulData)$p.value
	faithfulDipClock = faithfulDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsDip<-dip.test(carsData)$p.value
	carsDipClock = carsDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverDip<-dip.test(beaverData)$p.value
	beaverDipClock = beaverDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestDip<-dip.test(usArrests)$p.value
	USArrestDipClock = usArrestsClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudeDip<-dip.test(attitudeData)$p.value
	attitudeDipClock = attitudeDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisDip<-dip.test(irisData)$p.value
	irisDipClock = irisDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissDip<-dip.test(swissData)$p.value
	swissDipClock = swissDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treeDip<-dip.test(treeData)$p.value
	treeDipClock = treeDataClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingDip<-dip.test(usJudgeRatings)$p.value
	USJudgeRatingDipClock = usJudgeRatingsClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverDip<-dip.test(riverData)$p.value
	riverDipClock = riverDataClock + as.numeric(Sys.time()-checkpoint)
	

	#Classic Tests
	checkpoint<-Sys.time()
	faithfulClassicSilv<-silverman.test(as.matrix(faithful),1,adjust=TRUE)@p_value
	faithfulClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsClassicSilv<-silverman.test(as.matrix(cars),1,adjust=TRUE)@p_value
	carsClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverClassicSilv<-silverman.test(as.matrix(beaver1),1,adjust=TRUE)@p_value
	beaverClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestClassicSilv<-silverman.test(as.matrix(USArrests),1,adjust=TRUE)@p_value
	USArrestClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudeClassicSilv<-silverman.test(as.matrix(attitude),1,adjust=TRUE)@p_value
	attitudeClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisClassicSilv<-silverman.test(as.matrix(irisDataset()),1,adjust=TRUE)@p_value
	irisClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissClassicSilv<-silverman.test(as.matrix(swiss),1,adjust=TRUE)@p_value
	swissClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treeClassicSilv<-silverman.test(as.matrix(trees),1,adjust=TRUE)@p_value
	treeClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingClassicSilv<-silverman.test(as.matrix(USJudgeRatings),1,adjust=TRUE)@p_value
	USJudgeRatingClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverClassicSilv<-silverman.test(as.matrix(rivers),1,adjust=TRUE)@p_value
	riverClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsClassicSilv<-silverman.test(as.matrix(mtcars),1,adjust=TRUE)@p_value
	mtcarsClassicSilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	rockClassicSilv<-silverman.test(as.matrix(rock),1,adjust=TRUE)@p_value
	rockClassicSilvClock = as.numeric(Sys.time()-checkpoint)


	checkpoint<-Sys.time()
	rockClassicDip<-dip.test(as.matrix(rock))$p.value
	rockClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsClassicDip<-dip.test(as.matrix(mtcars))$p.value
	mtcarsClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	faithfulClassicDip<-dip.test(as.matrix(faithful))$p.value
	faithfulClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsClassicDip<-dip.test(as.matrix(cars))$p.value
	carsClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverClassicDip<-dip.test(as.matrix(beaver1))$p.value
	beaverClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestClassicDip<-dip.test(as.matrix(USArrests))$p.value
	USArrestClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudeClassicDip<-dip.test(as.matrix(attitude))$p.value
	attitudeClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisClassicDip<-dip.test(as.matrix(irisDataset()))$p.value
	irisClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissClassicDip<-dip.test(as.matrix(swiss))$p.value
	swissClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treeClassicDip<-dip.test(as.matrix(trees))$p.value
	treeClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingClassicDip<-dip.test(as.matrix(USJudgeRatings))$p.value
	USJudgeRatingClassicDipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverClassicDip<-dip.test(as.matrix(rivers))$p.value
	riverClassicDipClock = as.numeric(Sys.time()-checkpoint)


	#PCA Tests
	checkpoint<-Sys.time()
	faithfulPCASilv<-silverman.test(as.matrix(prcomp(faithful)$x[,1]),1,adjust=TRUE)@p_value
	faithfulPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsPCASilv<-silverman.test(as.matrix(prcomp(cars)$x[,1]),1,adjust=TRUE)@p_value
	carsPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverPCASilv<-silverman.test(as.matrix(prcomp(beaver1)$x[,1]),1,adjust=TRUE)@p_value
	beaverPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestPCASilv<-silverman.test(as.matrix(prcomp(USArrests)$x[,1]),1,adjust=TRUE)@p_value
	USArrestPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudePCASilv<-silverman.test(as.matrix(prcomp(attitude)$x[,1]),1,adjust=TRUE)@p_value
	attitudePCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisPCASilv<-silverman.test(as.matrix(prcomp(irisDataset())$x[,1]),1,adjust=TRUE)@p_value
	irisPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissPCASilv<-silverman.test(as.matrix(prcomp(swiss)$x[,1]),1,adjust=TRUE)@p_value
	swissPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treePCASilv<-silverman.test(as.matrix(prcomp(trees)$x[,1]),1,adjust=TRUE)@p_value
	treePCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingPCASilv<-silverman.test(as.matrix(prcomp(USJudgeRatings)$x[,1]),1,adjust=TRUE)@p_value
	USJudgeRatingPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverPCASilv<-silverman.test(as.matrix(prcomp(rivers)$x[,1]),1,adjust=TRUE)@p_value
	riverPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsPCASilv<-silverman.test(as.matrix(prcomp(mtcars)$x[,1]),1,adjust=TRUE)@p_value
	mtcarsPCASilvClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	rockPCASilv<-silverman.test(as.matrix(prcomp(rock)$x[,1]),1,adjust=TRUE)@p_value
	rockPCASilvClock = as.numeric(Sys.time()-checkpoint)
	
	checkpoint<-Sys.time()
	rockPCADip<-dip.test(as.matrix(prcomp(rock)$x[,1]))$p.value
	rockPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsPCADip<-dip.test(as.matrix(prcomp(mtcars)$x[,1]))$p.value
	mtcarsPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	faithfulPCADip<-dip.test(as.matrix(prcomp(faithful)$x[,1]))$p.value
	faithfulPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsPCADip<-dip.test(as.matrix(prcomp(cars)$x[,1]))$p.value
	carsPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverPCADip<-dip.test(as.matrix(prcomp(beaver1)$x[,1]))$p.value
	beaverPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestPCADip<-dip.test(as.matrix(prcomp(USArrests)$x[,1]))$p.value
	USArrestPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudePCADip<-dip.test(as.matrix(prcomp(attitude)$x[,1]))$p.value
	attitudePCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisPCADip<-dip.test(as.matrix(prcomp(irisDataset())$x[,1]))$p.value
	irisPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissPCADip<-dip.test(as.matrix(prcomp(swiss)$x[,1]))$p.value
	swissPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treePCADip<-dip.test(as.matrix(prcomp(trees)$x[,1]))$p.value
	treePCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingPCADip<-dip.test(as.matrix(prcomp(USJudgeRatings)$x[,1]))$p.value
	USJudgeRatingPCADipClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverPCADip<-dip.test(as.matrix(prcomp(rivers)$x[,1]))$p.value
	riverPCADipClock = as.numeric(Sys.time()-checkpoint)

#PRINCIPAL CURVE
	pcit=100
	
	checkpoint<-Sys.time()
	rockPrincurve<-principal.curve(as.matrix(rock), maxit=pcit)
	rockPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsPrincurve<-principal.curve(as.matrix(mtcars), maxit=pcit)
	mtcarsPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	faithfulPrincurve<-principal.curve(as.matrix(faithful), maxit=pcit)
	faithfulPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsPrincurve<-principal.curve(as.matrix(cars), maxit=pcit)
	carsPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverPrincurve<-principal.curve(as.matrix(beaver1), maxit=pcit)
	beaverPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestPrincurve<-principal.curve(as.matrix(USArrests), maxit=pcit)
	USArrestPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudePrincurve<-principal.curve(as.matrix(attitude), maxit=pcit)
	attitudePrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisPrincurve<-principal.curve(as.matrix(irisDataset()), maxit=pcit)
	irisPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissPrincurve<-principal.curve(as.matrix(swiss), maxit=pcit)
	swissPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treePrincurve<-principal.curve(as.matrix(trees), maxit=pcit)
	treePrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingPrincurve<-principal.curve(as.matrix(USJudgeRatings), maxit=pcit)
	USJudgeRatingPrincurveClock = as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverPrincurve<-principal.curve(as.matrix(rivers), maxit=pcit)
	riverPrincurveClock = as.numeric(Sys.time()-checkpoint)

#dip and silvertest
	checkpoint<-Sys.time()
	rockPrincurveDip<-dip.test(rockPrincurve$s[,1])$p.value
	rockPrincurveDipClock = rockPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsPrincurveDip<-dip.test(mtcarsPrincurve$s[,1])$p.value
	mtcarsPrincurveDipClock = mtcarsPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	faithfulPrincurveDip<-dip.test(faithfulPrincurve$s[,1])$p.value
	faithfulPrincurveDipClock = faithfulPrincurveClock  + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsPrincurveDip<-dip.test(carsPrincurve$s[,1])$p.value
	carsPrincurveDipClock = carsPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverPrincurveDip<-dip.test(beaverPrincurve$s[,1])$p.value
	beaverPrincurveDipClock = beaverPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestPrincurveDip<-dip.test(USArrestPrincurve$s[,1])$p.value
	USArrestPrincurveDipClock = USArrestPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudePrincurveDip<-dip.test(attitudePrincurve$s[,1])$p.value
	attitudePrincurveDipClock = attitudePrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisPrincurveDip<-dip.test(irisPrincurve$s[,1])$p.value
	irisPrincurveDipClock = irisPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissPrincurveDip<-dip.test(swissPrincurve$s[,1])$p.value
	swissPrincurveDipClock = swissPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treePrincurveDip<-dip.test(treePrincurve$s[,1])$p.value
	treePrincurveDipClock = treePrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingPrincurveDip<-dip.test(USJudgeRatingPrincurve$s[,1])$p.value
	USJudgeRatingPrincurveDipClock = USJudgeRatingPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverPrincurveDip<-dip.test(riverPrincurve$s[,1])$p.value
	riverPrincurveDipClock = riverPrincurveClock + as.numeric(Sys.time()-checkpoint)


	checkpoint<-Sys.time()
	rockPrincurveSilv<-silverman.test(rockPrincurve$s[,1],1,adjust=TRUE)@p_value
	rockPrincurveSilvClock = rockPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	mtcarsPrincurveSilv<-silverman.test(mtcarsPrincurve$s[,1],1,adjust=TRUE)@p_value
	mtcarsPrincurveSilvClock = mtcarsPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	faithfulPrincurveSilv<-silverman.test(faithfulPrincurve$s[,1],1,adjust=TRUE)@p_value
	faithfulPrincurveSilvClock = faithfulPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	carsPrincurveSilv<-silverman.test(carsPrincurve$s[,1],1,adjust=TRUE)@p_value
	carsPrincurveSilvClock = carsPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	beaverPrincurveSilv<-silverman.test(beaverPrincurve$s[,1],1,adjust=TRUE)@p_value
	beaverPrincurveSilvClock = beaverPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USArrestPrincurveSilv<-silverman.test(USArrestPrincurve$s[,1],1,adjust=TRUE)@p_value
	USArrestPrincurveSilvClock = USArrestPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	attitudePrincurveSilv<-silverman.test(attitudePrincurve$s[,1],1,adjust=TRUE)@p_value
	attitudePrincurveSilvClock = attitudePrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	irisPrincurveSilv<-silverman.test(irisPrincurve$s[,1],1,adjust=TRUE)@p_value
	irisPrincurveSilvClock = irisPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	swissPrincurveSilv<-silverman.test(swissPrincurve$s[,1],1,adjust=TRUE)@p_value
	swissPrincurveSilvClock = swissPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	treePrincurveSilv<-silverman.test(treePrincurve$s[,1],1,adjust=TRUE)@p_value
	treePrincurveSilvClock = treePrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	USJudgeRatingPrincurveSilv<-silverman.test(USJudgeRatingPrincurve$s[,1],1,adjust=TRUE)@p_value
	USJudgeRatingPrincurveSilvClock = USJudgeRatingPrincurveClock + as.numeric(Sys.time()-checkpoint)

	checkpoint<-Sys.time()
	riverPrincurveSilv<-silverman.test(riverPrincurve$s[,1],1,adjust=TRUE)@p_value
	riverPrincurveSilvClock = riverPrincurveClock + as.numeric(Sys.time()-checkpoint)

	
#trace convergence
	rockPrincurveConv<-rockPrincurve$converged
	mtcarsPrincurveConv<-mtcarsPrincurve$converged
	faithfulPrincurveConv<-faithfulPrincurve$converged
	carsPrincurveConv<-carsPrincurve$converged
	beaverPrincurveConv<-beaverPrincurve$converged
	USArrestPrincurveConv<-USArrestPrincurve$converged
	attitudePrincurveConv<-attitudePrincurve$converged
	irisPrincurveConv<-irisPrincurve$converged
	swissPrincurveConv<-swissPrincurve$converged
	treePrincurveConv<-treePrincurve$converged
	USJudgeRatingPrincurveConv<-USJudgeRatingPrincurve$converged
	riverPrincurveConv<-riverPrincurve$converged

	faithfulSample = nrow(faithful)/10
	carSample = nrow(cars)/10
	beaverSample = nrow(beaver1)/10
	USArrestSample = nrow(USArrests)/10
	attitudeSample = nrow(attitude)/10
	irisSample = nrow(irisDataset())/10
	swissSample = nrow(swiss)/10
	treeSample = nrow(trees)/10
	USJudgeSample = nrow(USJudgeRatings)/10
	riverSample = 14
	mtcarSample = nrow(mtcars)/10
	rockSample = nrow(rock)/10

	faithfulHopClock = 0
	carHopClock = 0
	beaverHopClock = 0
	usArrestHopClock = 0
	attitudeHopClock = 0
	irisHopClock = 0
	swissHopClock =0
	treeHopClock = 0
	usJudgeRatingHopClock = 0
	riverHopClock = 0
	mtcarsHopClock = 0
	rockHopClock = 0


	for(i in 1:it){
		checkpoint<-Sys.time()
		faithfulHop[i]<-hopkins(faithful, n = faithfulSample)$H
		faithfulHopClock = faithfulHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		carHop[i]<-hopkins(cars, n = carSample)$H
		carHopClock = carHopClock + as.numeric(Sys.time()-checkpoint)


		checkpoint<-Sys.time()
		beaverHop[i]<-hopkins(beaver1, n = beaverSample)$H
		beaverHopClock = beaverHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		usArrestHop[i]<-hopkins(USArrests, n = USArrestSample)$H
		usArrestHopClock = usArrestHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		attitudeHop[i]<-hopkins(attitude, n = attitudeSample)$H
		attitudeHopClock = attitudeHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		irisHop[i]<-hopkins(irisDataset(), n = irisSample)$H
		irisHopClock = irisHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		swissHop[i]<-hopkins(swiss, n = swissSample)$H
		swissHopClock = swissHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		treeHop[i]<-hopkins(trees, n = treeSample)$H
		treeHopClock = treeHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		usJudgeRatingHop[i]<-hopkins(USJudgeRatings, n = USJudgeSample)$H
		usJudgeRatingHopClock = usJudgeRatingHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		riverHop[i]<-hopkins(as.data.frame(rivers), n = riverSample)$H
		riverHopClock = riverHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		mtcarsHop[i]<-hopkins(mtcars, n = mtcarSample)$H
		mtcarsHopClock = mtcarsHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		rockHop[i]<-hopkins(rock, n = rockSample)$H
		rockHopClock = rockHopClock + as.numeric(Sys.time()-checkpoint)

	}

#67
	result<-matrix(,12,15)
	timer<-matrix(,12,9)


	timer[1,1]<-round(carsDipClock,4)
	timer[1,2]<-round(carsSilvClock,4)
	timer[1,3]<-round(carHopClock/it,4)
	timer[1,4]<-round(carsClassicDipClock,4)
	timer[1,5]<-round(carsClassicSilvClock,4)
	timer[1,6]<-round(carsPCADipClock,4)
	timer[1,7]<-round(carsPCASilvClock,4)
	timer[1,8]<-round(carsPrincurveDipClock,4)
	timer[1,9]<-round(carsPrincurveSilvClock,4)
	timer[2,1]<-round(attitudeDipClock,4)
	timer[2,2]<-round(attitudeSilvClock,4)
	timer[2,3]<-round(attitudeHopClock/it,4)
	timer[2,4]<-round(attitudeClassicDipClock,4)
	timer[2,5]<-round(attitudeClassicSilvClock,4)
	timer[2,6]<-round(attitudePCADipClock,4)
	timer[2,7]<-round(attitudePCASilvClock,4)
	timer[2,8]<-round(attitudePrincurveDipClock,4)
	timer[2,9]<-round(attitudePrincurveSilvClock,4)
	timer[3,1]<-round(USArrestDipClock,4)
	timer[3,2]<-round(USArrestSilvClock,4)
	timer[3,3]<-round(usArrestHopClock/it,4)
	timer[3,4]<-round(USArrestClassicDipClock,4)
	timer[3,5]<-round(USArrestClassicSilvClock,4)
	timer[3,6]<-round(USArrestPCADipClock,4)
	timer[3,7]<-round(USArrestPCASilvClock,4)
	timer[3,8]<-round(USArrestPrincurveDipClock,4)
	timer[3,9]<-round(USArrestPrincurveSilvClock,4)
	timer[4,1]<-round(treeDipClock,4)
	timer[4,2]<-round(treeSilvClock,4)
	timer[4,3]<-round(treeHopClock/it,4)
	timer[4,4]<-round(treeClassicDipClock,4)
	timer[4,5]<-round(treeClassicSilvClock,4)
	timer[4,6]<-round(treePCADipClock,4)
	timer[4,7]<-round(treePCASilvClock,4)
	timer[4,8]<-round(treePrincurveDipClock,4)
	timer[4,9]<-round(treePrincurveSilvClock,4)
	timer[5,1]<-round(USJudgeRatingDipClock,4)
	timer[5,2]<-round(USJudgeRatingSilvClock,4)
	timer[5,3]<-round(usJudgeRatingHopClock/it,4)
	timer[5,4]<-round(USJudgeRatingClassicDipClock,4)
	timer[5,5]<-round(USJudgeRatingClassicSilvClock,4)
	timer[5,6]<-round(USJudgeRatingPCADipClock,4)
	timer[5,7]<-round(USJudgeRatingPCASilvClock,4)
	timer[5,8]<-round(USJudgeRatingPrincurveDipClock,4)
	timer[5,9]<-round(USJudgeRatingPrincurveSilvClock,4)
	timer[6,1]<-round(riverDipClock,4)
	timer[6,2]<-round(riverSilvClock,4)
	timer[6,3]<-round(riverHopClock/it,4)
	timer[6,4]<-round(riverClassicDipClock,4)
	timer[6,5]<-round(riverClassicSilvClock,4)
	timer[6,6]<-round(riverPCADipClock,4)
	timer[6,7]<-round(riverPCASilvClock,4)
	timer[6,8]<-round(riverPrincurveDipClock,4)
	timer[6,9]<-round(riverPrincurveSilvClock,4)
	timer[7,1]<-round(faithfulDipClock,4)
	timer[7,2]<-round(faithfulSilvClock,4)
	timer[7,3]<-round(faithfulHopClock/it,4)
	timer[7,4]<-round(faithfulClassicDipClock,4)
	timer[7,5]<-round(faithfulClassicSilvClock,4)
	timer[7,6]<-round(faithfulPCADipClock,4)
	timer[7,7]<-round(faithfulPCASilvClock,4)
	timer[7,8]<-round(faithfulPrincurveDipClock,4)
	timer[7,9]<-round(faithfulPrincurveSilvClock,4)
	timer[8,1]<-round(beaverDipClock,4)
	timer[8,2]<-round(beaverSilvClock,4)
	timer[8,3]<-round(beaverHopClock/it,4)
	timer[8,4]<-round(beaverClassicDipClock,4)
	timer[8,5]<-round(beaverClassicSilvClock,4)
	timer[8,6]<-round(beaverPCADipClock,4)
	timer[8,7]<-round(beaverPCASilvClock,4)
	timer[8,8]<-round(beaverPrincurveDipClock,4)
	timer[8,9]<-round(beaverPrincurveSilvClock,4)
	timer[9,1]<-round(irisDipClock,4)
	timer[9,2]<-round(irisSilvClock,4)
	timer[9,3]<-round(irisHopClock/it,4)
	timer[9,4]<-round(irisClassicDipClock,4)
	timer[9,5]<-round(irisClassicSilvClock,4)
	timer[9,6]<-round(irisPCADipClock,4)
	timer[9,7]<-round(irisPCASilvClock,4)
	timer[9,8]<-round(irisPrincurveDipClock,4)
	timer[9,9]<-round(irisPrincurveSilvClock,4)
	timer[10,1]<-round(swissDipClock,4)
	timer[10,2]<-round(swissSilvClock,4)
	timer[10,3]<-round(swissHopClock/it,4)
	timer[10,4]<-round(swissClassicDipClock,4)
	timer[10,5]<-round(swissClassicSilvClock,4)	
	timer[10,6]<-round(swissPCADipClock,4)
	timer[10,7]<-round(swissPCASilvClock,4)
	timer[10,8]<-round(swissPrincurveDipClock,4)
	timer[10,9]<-round(swissPrincurveSilvClock,4)
	timer[11,1]<-round(mtcarsDipClock,4)
	timer[11,2]<-round(mtcarsSilvClock,4)
	timer[11,3]<-round(mtcarsHopClock/it,4)
	timer[11,4]<-round(mtcarsClassicDipClock,4)
	timer[11,5]<-round(mtcarsClassicSilvClock,4)	
	timer[11,6]<-round(mtcarsPCADipClock,4)
	timer[11,7]<-round(mtcarsPCASilvClock,4)
	timer[11,8]<-round(mtcarsPrincurveDipClock,4)
	timer[11,9]<-round(mtcarsPrincurveSilvClock,4)
	timer[12,1]<-round(rockDipClock,4)
	timer[12,2]<-round(rockSilvClock,4)
	timer[12,3]<-round(rockHopClock/it,4)
	timer[12,4]<-round(rockClassicDipClock,4)
	timer[12,5]<-round(rockClassicSilvClock,4)	
	timer[12,6]<-round(rockPCADipClock,4)
	timer[12,7]<-round(rockPCASilvClock,4)
	timer[12,8]<-round(rockPrincurveDipClock,4)
	timer[12,9]<-round(rockPrincurveSilvClock,4)



	colnames(timer)<-c("Dip","Silverman","Hopkins","Classic Dip","Classic Silv.","PCA Dip","PCA Silv.","Pcurv.Dip","Pcurv.Silv")
	rownames(timer)<-c("cars","attitude","USArrests","trees","USJudgeRatings","rivers","faithful",
					"beaver1","iris","swiss","mtcars","rock")

	print(timer)


#

	result[1,1]<-round(carsDip,4)
	result[1,2]<-round(carsSilv,4)
	result[1,3]<-round(mean(carHop),4)
	result[1,4]<-round(sum(carHop<0.25)/it,4)
	result[1,5]<-round(sum(carHop<qbeta(0.025, carSample, carSample))/it,4)
	result[1,6]<-round(sum(carHop<qbeta(0.05, carSample, carSample))/it,4)
	result[1,7]<-round(sum(carHop>qbeta(0.95, carSample, carSample))/it,4)
	result[1,8]<-round(sum(carHop>qbeta(0.975, carSample, carSample))/it,4)
	result[1,9]<-round(carsClassicDip,4)
	result[1,10]<-round(carsClassicSilv,4)
	result[1,11]<-round(carsPCADip,4)
	result[1,12]<-round(carsPCASilv,4)
	result[1,13]<-carsPrincurveConv
	result[1,14]<-round(carsPrincurveDip,4)
	result[1,15]<-round(carsPrincurveSilv,4)
	result[2,1]<-round(attitudeDip,4)
	result[2,2]<-round(attitudeSilv,4)
	result[2,3]<-round(mean(attitudeHop),4)
	result[2,4]<-round(sum(attitudeHop<0.25)/it,4)
	result[2,5]<-round(sum(attitudeHop<qbeta(0.025, attitudeSample, attitudeSample))/it,4)
	result[2,6]<-round(sum(attitudeHop<qbeta(0.05, attitudeSample, attitudeSample))/it,4)
	result[2,7]<-round(sum(attitudeHop>qbeta(0.95, attitudeSample, attitudeSample))/it,4)
	result[2,8]<-round(sum(attitudeHop>qbeta(0.975, attitudeSample, attitudeSample))/it,4)
	result[2,9]<-round(attitudeClassicDip,4)
	result[2,10]<-round(attitudeClassicSilv,4)
	result[2,11]<-round(attitudePCADip,4)
	result[2,12]<-round(attitudePCASilv,4)
	result[2,13]<-attitudePrincurveConv
	result[2,14]<-round(attitudePrincurveDip,4)
	result[2,15]<-round(attitudePrincurveSilv,4)
	result[3,1]<-round(USArrestDip,4)
	result[3,2]<-round(USArrestSilv,4)
	result[3,3]<-round(mean(usArrestHop),4)
	result[3,4]<-round(sum(usArrestHop<0.25)/it,4)
	result[3,5]<-round(sum(usArrestHop<qbeta(0.025, USArrestSample, USArrestSample))/it,4)
	result[3,6]<-round(sum(usArrestHop<qbeta(0.05, USArrestSample, USArrestSample))/it,4)
	result[3,7]<-round(sum(usArrestHop>qbeta(0.95, USArrestSample, USArrestSample))/it,4)
	result[3,8]<-round(sum(usArrestHop>qbeta(0.975, USArrestSample, USArrestSample))/it,4)
	result[3,9]<-round(USArrestClassicDip,4)
	result[3,10]<-round(USArrestClassicSilv,4)
	result[3,11]<-round(USArrestPCADip,4)
	result[3,12]<-round(USArrestPCASilv,4)
	result[3,13]<-USArrestPrincurveConv
	result[3,14]<-round(USArrestPrincurveDip,4)
	result[3,15]<-round(USArrestPrincurveSilv,4)
	result[4,1]<-round(treeDip,4)
	result[4,2]<-round(treeSilv,4)
	result[4,3]<-round(mean(treeHop),4)
	result[4,4]<-round(sum(treeHop<0.25)/it,4)
	result[4,5]<-round(sum(treeHop<qbeta(0.025, treeSample, treeSample))/it,4)
	result[4,6]<-round(sum(treeHop<qbeta(0.05, treeSample, treeSample))/it,4)
	result[4,7]<-round(sum(treeHop>qbeta(0.95, treeSample, treeSample))/it,4)
	result[4,8]<-round(sum(treeHop>qbeta(0.975, treeSample, treeSample))/it,4)
	result[4,9]<-round(treeClassicDip,4)
	result[4,10]<-round(treeClassicSilv,4)
	result[4,11]<-round(treePCADip,4)
	result[4,12]<-round(treePCASilv,4)
	result[4,13]<-treePrincurveConv
	result[4,14]<-round(treePrincurveDip,4)
	result[4,15]<-round(treePrincurveSilv,4)
	result[5,1]<-round(USJudgeRatingDip,4)
	result[5,2]<-round(USJudgeRatingSilv,4)
	result[5,3]<-round(mean(usJudgeRatingHop),4)
	result[5,4]<-round(sum(usJudgeRatingHop<0.25)/it,4)
	result[5,5]<-round(sum(usJudgeRatingHop<qbeta(0.025, USJudgeSample, USJudgeSample))/it,4)
	result[5,6]<-round(sum(usJudgeRatingHop<qbeta(0.05, USJudgeSample, USJudgeSample))/it,4)
	result[5,7]<-round(sum(usJudgeRatingHop>qbeta(0.95, USJudgeSample, USJudgeSample))/it,4)
	result[5,8]<-round(sum(usJudgeRatingHop>qbeta(0.975, USJudgeSample, USJudgeSample))/it,4)
	result[5,9]<-round(USJudgeRatingClassicDip,4)
	result[5,10]<-round(USJudgeRatingClassicSilv,4)
	result[5,11]<-round(USJudgeRatingPCADip,4)
	result[5,12]<-round(USJudgeRatingPCASilv,4)
	result[5,13]<-USJudgeRatingPrincurveConv
	result[5,14]<-round(USJudgeRatingPrincurveDip,4)
	result[5,15]<-round(USJudgeRatingPrincurveSilv,4)
	result[6,1]<-round(riverDip,4)
	result[6,2]<-round(riverSilv,4)
	result[6,3]<-round(mean(riverHop),4)
	result[6,4]<-round(sum(riverHop<0.25)/it,4)
	result[6,5]<-round(sum(riverHop<qbeta(0.025, riverSample, riverSample))/it,4)
	result[6,6]<-round(sum(riverHop<qbeta(0.05, riverSample, riverSample))/it,4)
	result[6,7]<-round(sum(riverHop>qbeta(0.95, riverSample, riverSample))/it,4)
	result[6,8]<-round(sum(riverHop>qbeta(0.975, riverSample, riverSample))/it,4)
	result[6,9]<-round(riverClassicDip,4)
	result[6,10]<-round(riverClassicSilv,4)
	result[6,11]<-round(riverPCADip,4)
	result[6,12]<-round(riverPCASilv,4)
	result[6,13]<-riverPrincurveConv
	result[6,14]<-round(riverPrincurveDip,4)
	result[6,15]<-round(riverPrincurveSilv,4)
	result[7,1]<-round(faithfulDip,4)
	result[7,2]<-round(faithfulSilv,4)
	result[7,3]<-round(mean(faithfulHop),4)
	result[7,4]<-round(sum(faithfulHop<0.25)/it,4)
	result[7,5]<-round(sum(faithfulHop<qbeta(0.025, faithfulSample, faithfulSample))/it,4)
	result[7,6]<-round(sum(faithfulHop<qbeta(0.05, faithfulSample, faithfulSample))/it,4)
	result[7,7]<-round(sum(faithfulHop>qbeta(0.95, faithfulSample, faithfulSample))/it,4)
	result[7,8]<-round(sum(faithfulHop>qbeta(0.975, faithfulSample, faithfulSample))/it,4)
	result[7,9]<-round(faithfulClassicDip,4)
	result[7,10]<-round(faithfulClassicSilv,4)
	result[7,11]<-round(faithfulPCADip,4)
	result[7,12]<-round(faithfulPCASilv,4)
	result[7,13]<-faithfulPrincurveConv
	result[7,14]<-round(faithfulPrincurveDip,4)
	result[7,15]<-round(faithfulPrincurveSilv,4)
	result[8,1]<-round(beaverDip,4)
	result[8,2]<-round(beaverSilv,4)
	result[8,3]<-round(mean(beaverHop),4)
	result[8,4]<-round(sum(beaverHop<0.25)/it,4)
	result[8,5]<-round(sum(beaverHop<qbeta(0.025, beaverSample, beaverSample))/it,4)
	result[8,6]<-round(sum(beaverHop<qbeta(0.05, beaverSample, beaverSample))/it,4)
	result[8,7]<-round(sum(beaverHop>qbeta(0.95, beaverSample, beaverSample))/it,4)
	result[8,8]<-round(sum(beaverHop>qbeta(0.975, beaverSample, beaverSample))/it,4)
	result[8,9]<-round(beaverClassicDip,4)
	result[8,10]<-round(beaverClassicSilv,4)
	result[8,11]<-round(beaverPCADip,4)
	result[8,12]<-round(beaverPCASilv,4)
	result[8,13]<-beaverPrincurveConv
	result[8,14]<-round(beaverPrincurveDip,4)
	result[8,15]<-round(beaverPrincurveSilv,4)
	result[9,1]<-round(irisDip,4)
	result[9,2]<-round(irisSilv,4)
	result[9,3]<-round(mean(irisHop),4)
	result[9,4]<-round(sum(irisHop<0.25)/it,4)
	result[9,5]<-round(sum(irisHop<qbeta(0.025, irisSample, irisSample))/it,4)
	result[9,6]<-round(sum(irisHop<qbeta(0.05, irisSample, irisSample))/it,4)
	result[9,7]<-round(sum(irisHop>qbeta(0.95, irisSample, irisSample))/it,4)
	result[9,8]<-round(sum(irisHop>qbeta(0.975, irisSample, irisSample))/it,4)
	result[9,9]<-round(irisClassicDip,4)
	result[9,10]<-round(irisClassicSilv,4)
	result[9,11]<-round(irisPCADip,4)
	result[9,12]<-round(irisPCASilv,4)
	result[9,13]<-irisPrincurveConv
	result[9,14]<-round(irisPrincurveDip,4)
	result[9,15]<-round(irisPrincurveSilv,4)
	result[10,1]<-round(swissDip,4)
	result[10,2]<-round(swissSilv,4)
	result[10,3]<-round(mean(swissHop),4)
	result[10,4]<-round(sum(swissHop<0.25)/it,4)
	result[10,5]<-round(sum(swissHop<qbeta(0.025, swissSample, swissSample))/it,4)
	result[10,6]<-round(sum(swissHop<qbeta(0.05, swissSample, swissSample))/it,4)
	result[10,7]<-round(sum(swissHop>qbeta(0.95, swissSample, swissSample))/it,4)
	result[10,8]<-round(sum(swissHop>qbeta(0.975, swissSample, swissSample))/it,4)
	result[10,9]<-round(swissClassicDip,4)
	result[10,10]<-round(swissClassicSilv,4)	
	result[10,11]<-round(swissPCADip,4)
	result[10,12]<-round(swissPCASilv,4)
	result[10,13]<-swissPrincurveConv
	result[10,14]<-round(swissPrincurveDip,4)
	result[10,15]<-round(swissPrincurveSilv,4)
	result[11,1]<-round(mtcarsDip,4)
	result[11,2]<-round(mtcarsSilv,4)
	result[11,3]<-round(mean(mtcarsHop),4)
	result[11,4]<-round(sum(mtcarsHop<0.25)/it,4)
	result[11,5]<-round(sum(mtcarsHop<qbeta(0.025, mtcarSample, mtcarSample))/it,4)
	result[11,6]<-round(sum(mtcarsHop<qbeta(0.05, mtcarSample, mtcarSample))/it,4)
	result[11,7]<-round(sum(mtcarsHop>qbeta(0.95, mtcarSample, mtcarSample))/it,4)
	result[11,8]<-round(sum(mtcarsHop>qbeta(0.975, mtcarSample, mtcarSample))/it,4)
	result[11,9]<-round(mtcarsClassicDip,4)
	result[11,10]<-round(mtcarsClassicSilv,4)	
	result[11,11]<-round(mtcarsPCADip,4)
	result[11,12]<-round(mtcarsPCASilv,4)
	result[11,13]<-mtcarsPrincurveConv
	result[11,14]<-round(mtcarsPrincurveDip,4)
	result[11,15]<-round(mtcarsPrincurveSilv,4)
	result[12,1]<-round(rockDip,4)
	result[12,2]<-round(rockSilv,4)
	result[12,3]<-round(mean(rockHop),4)
	result[12,4]<-round(sum(rockHop<0.25)/it,4)
	result[12,5]<-round(sum(rockHop<qbeta(0.025, rockSample, rockSample))/it,4)
	result[12,6]<-round(sum(rockHop<qbeta(0.05, rockSample, rockSample))/it,4)
	result[12,7]<-round(sum(rockHop>qbeta(0.95, rockSample, rockSample))/it,4)
	result[12,8]<-round(sum(rockHop>qbeta(0.975, rockSample, rockSample))/it,4)
	result[12,9]<-round(rockClassicDip,4)
	result[12,10]<-round(rockClassicSilv,4)	
	result[12,11]<-round(rockPCADip,4)
	result[12,12]<-round(rockPCASilv,4)
	result[12,13]<-rockPrincurveConv
	result[12,14]<-round(rockPrincurveDip,4)
	result[12,15]<-round(rockPrincurveSilv,4)



	colnames(result)<-c("Dip","Silverman","Hopkins","Hop<0.25","Hop<qbeta.025","Hop<qbeta.05","Hop>qbeta.95","Hop>qbeta.975",
					"Classic Dip","Classic Silv.","PCA Dip","PCA Silv.","Pcurv.Conv","Pcurv.Dip","Pcurv.Silv")
	rownames(result)<-c("cars","attitude","USArrests","trees","USJudgeRatings","rivers","faithful",
					"beaver1","iris","swiss","mtcars","rock")

	write.table(timer,sep="&",file="realdatatimer.txt")
	write.table(result,sep="&",file="realdataresult.txt")

	return(result)
}

generatedData<-function(it){
	clock = 0;

	naomiSilv=rep(NA,it)
	naomiDip=rep(NA,it)
	naomiHop=rep(NA,it)
	naomiClassicDip=rep(NA,it)
	naomiClassicSilv=rep(NA,it)
	naomiPCASilv=rep(NA,it)
	naomiPCADip=rep(NA,it)

	singleSilv=rep(NA,it)
	singleDip=rep(NA,it)
	singleHop=rep(NA,it)
	singleClassicDip=rep(NA,it)
	singleClassicSilv=rep(NA,it)
	singlePCASilv=rep(NA,it)
	singlePCADip=rep(NA,it)
	single3DSilv=rep(NA,it)
	single3DDip=rep(NA,it)
	single3DHop=rep(NA,it)
	single3DClassicSilv=rep(NA,it)
	single3DClassicDip=rep(NA,it)
	single3DPCASilv=rep(NA,it)
	single3DPCADip=rep(NA,it)
	single10DSilv=rep(NA,it)
	single10DDip=rep(NA,it)
	single10DHop=rep(NA,it)
	single10DClassicSilv=rep(NA,it)
	single10DClassicDip=rep(NA,it)
	single10DPCASilv=rep(NA,it)
	single10DPCADip=rep(NA,it)
	singleOutlierSilv=rep(NA,it)
	singleOutlierDip=rep(NA,it)
	singleOutlierHop=rep(NA,it)
	singleOutlierClassicSilv=rep(NA,it)
	singleOutlierClassicDip=rep(NA,it)
	singleOutlierPCASilv=rep(NA,it)
	singleOutlierPCADip=rep(NA,it)
	largeOutlierSilv=rep(NA,it)
	largeOutlierDip=rep(NA,it)
	largeOutlierHop=rep(NA,it)
	largeOutlierClassicSilv=rep(NA,it)
	largeOutlierClassicDip=rep(NA,it)
	largeOutlierPCASilv=rep(NA,it)
	largeOutlierPCADip=rep(NA,it)
	single3OutliersSilv=rep(NA,it)
	single3OutliersDip=rep(NA,it)
	single3OutliersHop=rep(NA,it)
	single3OutliersClassicSilv=rep(NA,it)
	single3OutliersClassicDip=rep(NA,it)
	single3OutliersPCASilv=rep(NA,it)
	single3OutliersPCADip=rep(NA,it)	
	twoSepSilv=rep(NA,it)
	twoSepDip=rep(NA,it)
	twoSepHop=rep(NA,it)
	twoSepClassicSilv=rep(NA,it)
	twoSepClassicDip=rep(NA,it)
	twoSepPCASilv=rep(NA,it)
	twoSepPCADip=rep(NA,it)
	threeSepSilv=rep(NA,it)
	threeSepDip=rep(NA,it)
	threeSepHop=rep(NA,it)
	threeSepClassicSilv=rep(NA,it)
	threeSepClassicDip=rep(NA,it)
	threeSepPCASilv=rep(NA,it)
	threeSepPCADip=rep(NA,it)	
	threeSep3DSilv=rep(NA,it)
	threeSep3DDip=rep(NA,it)
	threeSep3DHop=rep(NA,it)
	threeSep3DClassicSilv=rep(NA,it)
	threeSep3DClassicDip=rep(NA,it)
	threeSep3DPCASilv=rep(NA,it)
	threeSep3DPCADip=rep(NA,it)
	threeCloseSilv=rep(NA,it)
	threeCloseDip=rep(NA,it)
	threeCloseHop=rep(NA,it)
	threeCloseClassicSilv=rep(NA,it)
	threeCloseClassicDip=rep(NA,it)
	threeClosePCASilv=rep(NA,it)
	threeClosePCADip=rep(NA,it)
	threeNoisySilv=rep(NA,it)
	threeNoisyDip=rep(NA,it)
	threeNoisyHop=rep(NA,it)
	threeNoisyClassicSilv=rep(NA,it)
	threeNoisyClassicDip=rep(NA,it)
	threeNoisyPCASilv=rep(NA,it)
	threeNoisyPCADip=rep(NA,it)
	threeDiffDiameterSilv=rep(NA,it)
	threeDiffDiameterDip=rep(NA,it)
	threeDiffDiameterHop=rep(NA,it)
	threeDiffDiameterClassicSilv=rep(NA,it)
	threeDiffDiameterClassicDip=rep(NA,it)
	threeDiffDiameterPCASilv=rep(NA,it)
	threeDiffDiameterPCADip=rep(NA,it)
	threeDiffDensitySilv=rep(NA,it)
	threeDiffDensityDip=rep(NA,it)
	threeDiffDensityHop=rep(NA,it)
	threeDiffDensityClassicSilv=rep(NA,it)
	threeDiffDensityClassicDip=rep(NA,it)
	threeDiffDensityPCASilv=rep(NA,it)
	threeDiffDensityPCADip=rep(NA,it)
	twoSep10DSilv=rep(NA,it)
	twoSep10DDip=rep(NA,it)
	twoSep10DHop=rep(NA,it)
	twoSep10DClassicSilv=rep(NA,it)
	twoSep10DClassicDip=rep(NA,it)
	twoSep10DPCASilv=rep(NA,it)
	twoSep10DPCADip=rep(NA,it)
	fourSep10DSilv=rep(NA,it)
	fourSep10DDip=rep(NA,it)
	fourSep10DHop=rep(NA,it)
	fourSep10DClassicSilv=rep(NA,it)
	fourSep10DClassicDip=rep(NA,it)
	fourSep10DPCASilv=rep(NA,it)
	fourSep10DPCADip=rep(NA,it)
	tenSeparatedSilv=rep(NA,it)
	tenSeparatedDip=rep(NA,it)
	tenSeparatedHop=rep(NA,it)
	tenSeparatedClassicSilv=rep(NA,it)
	tenSeparatedClassicDip=rep(NA,it)
	tenSeparatedPCASilv=rep(NA,it)
	tenSeparatedPCADip=rep(NA,it)
	tenCloseSilv=rep(NA,it)
	tenCloseDip=rep(NA,it)
	tenCloseHop=rep(NA,it)
	tenCloseClassicSilv=rep(NA,it)
	tenCloseClassicDip=rep(NA,it)
	tenClosePCASilv=rep(NA,it)
	tenClosePCADip=rep(NA,it)
	fiveCloseSilv=rep(NA,it)
	fiveCloseDip=rep(NA,it)
	fiveCloseHop=rep(NA,it)
	fiveCloseClassicSilv=rep(NA,it)
	fiveCloseClassicDip=rep(NA,it)
	fiveClosePCASilv=rep(NA,it)
	fiveClosePCADip=rep(NA,it)
	sevenCloseSilv=rep(NA,it)
	sevenCloseDip=rep(NA,it)
	sevenCloseHop=rep(NA,it)
	sevenCloseClassicSilv=rep(NA,it)
	sevenCloseClassicDip=rep(NA,it)
	sevenClosePCASilv=rep(NA,it)
	sevenClosePCADip=rep(NA,it)
	sevenCloseDenseSilv=rep(NA,it)
	sevenCloseDenseDip=rep(NA,it)
	sevenCloseDenseHop=rep(NA,it)
	sevenCloseDenseClassicSilv=rep(NA,it)
	sevenCloseDenseClassicDip=rep(NA,it)
	sevenCloseDensePCASilv=rep(NA,it)
	sevenCloseDensePCADip=rep(NA,it)
	parallelLinesSilv=rep(NA,it)
	parallelLinesDip=rep(NA,it)
	parallelLinesHop=rep(NA,it)
	parallelLinesClassicSilv=rep(NA,it)
	parallelLinesClassicDip=rep(NA,it)
	parallelLinesPCASilv=rep(NA,it)
	parallelLinesPCADip=rep(NA,it)
	concCirclesSilv=rep(NA,it)
	concCirclesDip=rep(NA,it)
	concCirclesHop=rep(NA,it)
	concCirclesClassicSilv=rep(NA,it)
	concCirclesClassicDip=rep(NA,it)
	concCirclesPCASilv=rep(NA,it)
	concCirclesPCADip=rep(NA,it)
	threeConcCirclesSilv=rep(NA,it)
	threeConcCirclesDip=rep(NA,it)
	threeConcCirclesHop=rep(NA,it)
	threeConcCirclesClassicSilv=rep(NA,it)
	threeConcCirclesClassicDip=rep(NA,it)
	threeConcCirclesPCASilv=rep(NA,it)
	threeConcCirclesPCADip=rep(NA,it)
	oneTclust5dfSilv=rep(NA,it)
	oneTclust5dfDip=rep(NA,it)
	oneTclust5dfHop=rep(NA,it)
	oneTclust5dfClassicSilv=rep(NA,it)
	oneTclust5dfClassicDip=rep(NA,it)
	oneTclust5dfPCASilv=rep(NA,it)
	oneTclust5dfPCADip=rep(NA,it)
	twoTclust5dfSilv=rep(NA,it)
	twoTclust5dfDip=rep(NA,it)
	twoTclust5dfHop=rep(NA,it)
	twoTclust5dfClassicSilv=rep(NA,it)
	twoTclust5dfClassicDip=rep(NA,it)
	twoTclust5dfPCASilv=rep(NA,it)
	twoTclust5dfPCADip=rep(NA,it)
	oneTclust10dfSilv=rep(NA,it)
	oneTclust10dfDip=rep(NA,it)
	oneTclust10dfHop=rep(NA,it)
	oneTclust10dfClassicSilv=rep(NA,it)
	oneTclust10dfClassicDip=rep(NA,it)
	oneTclust10dfPCASilv=rep(NA,it)
	oneTclust10dfPCADip=rep(NA,it)
	twoTclust10dfSilv=rep(NA,it)
	twoTclust10dfDip=rep(NA,it)
	twoTclust10dfHop=rep(NA,it)
	twoTclust10dfClassicSilv=rep(NA,it)
	twoTclust10dfClassicDip=rep(NA,it)
	twoTclust10dfPCASilv=rep(NA,it)
	twoTclust10dfPCADip=rep(NA,it)
	oneTclust15dfSilv=rep(NA,it)
	oneTclust15dfDip=rep(NA,it)
	oneTclust15dfHop=rep(NA,it)
	oneTclust15dfClassicSilv=rep(NA,it)
	oneTclust15dfClassicDip=rep(NA,it)
	oneTclust15dfPCASilv=rep(NA,it)
	oneTclust15dfPCADip=rep(NA,it)
	twoTclust15dfSilv=rep(NA,it)
	twoTclust15dfDip=rep(NA,it)
	twoTclust15dfHop=rep(NA,it)
	twoTclust15dfClassicSilv=rep(NA,it)
	twoTclust15dfClassicDip=rep(NA,it)
	twoTclust15dfPCASilv=rep(NA,it)
	twoTclust15dfPCADip=rep(NA,it)
	singleClust50DSilv=rep(NA,it)
	singleClust50DDip=rep(NA,it)
	singleClust50DHop=rep(NA,it)
	singleClust50DClassicDip=rep(NA,it)
	singleClust50DClassicSilv=rep(NA,it)
	singleClust50DPCASilv=rep(NA,it)
	singleClust50DPCADip=rep(NA,it)
	fiveConcCircleSilv=rep(NA,it)
	fiveConcCircleDip=rep(NA,it)
	fiveConcCircleHop=rep(NA,it)
	fiveConcCircleClassicDip=rep(NA,it)
	fiveConcCircleClassicSilv=rep(NA,it)
	fiveConcCirclePCASilv=rep(NA,it)
	fiveConcCirclePCADip=rep(NA,it)		
	twoCloseClust50DSilv=rep(NA,it)
	twoCloseClust50DDip=rep(NA,it)
	twoCloseClust50DHop=rep(NA,it)
	twoCloseClust50DClassicDip=rep(NA,it)
	twoCloseClust50DClassicSilv=rep(NA,it)
	twoCloseClust50DPCASilv=rep(NA,it)
	twoCloseClust50DPCADip=rep(NA,it)
	twoOverlapClust50DSilv=rep(NA,it)
	twoOverlapClust50DDip=rep(NA,it)
	twoOverlapClust50DHop=rep(NA,it)
	twoOverlapClust50DClassicDip=rep(NA,it)
	twoOverlapClust50DClassicSilv=rep(NA,it)
	twoOverlapClust50DPCASilv=rep(NA,it)
	twoOverlapClust50DPCADip=rep(NA,it)
	

	naomiPrincurveConv=rep(NA,it)
	singleClust50DPrincurveConv=rep(NA,it)
	fiveConcCirclePrincurveConv=rep(NA,it)
	singlePrincurveConv=rep(NA,it)
	single3DPrincurveConv=rep(NA,it)
	single10DPrincurveConv=rep(NA,it)
	singleOutlierPrincurveConv=rep(NA,it)
	largeOutlierPrincurveConv=rep(NA,it)
	single3OutliersPrincurveConv=rep(NA,it)
	twoSepPrincurveConv=rep(NA,it)
	threeSepPrincurveConv=rep(NA,it)
	threeSep3DPrincurveConv=rep(NA,it)
	threeClosePrincurveConv=rep(NA,it)
	threeNoisyPrincurveConv=rep(NA,it)
	threeDiffDiameterPrincurveConv=rep(NA,it)
	threeDiffDensityPrincurveConv=rep(NA,it)
	twoSep10DPrincurveConv=rep(NA,it)
	fourSep10DPrincurveConv=rep(NA,it)
	tenSeparatedPrincurveConv=rep(NA,it)
	tenClosePrincurveConv=rep(NA,it)
	sevenCloseDensePrincurveConv=rep(NA,it)
	fiveClosePrincurveConv=rep(NA,it)
	sevenClosePrincurveConv=rep(NA,it)
	parallelLinesPrincurveConv=rep(NA,it)
	concCirclesPrincurveConv=rep(NA,it)
	threeConcCirclesPrincurveConv=rep(NA,it)
	oneTclust10dfPrincurveConv=rep(NA,it)
	twoTclust10dfPrincurveConv=rep(NA,it)
	oneTclust5dfPrincurveConv=rep(NA,it)
	twoTclust5dfPrincurveConv=rep(NA,it)
	oneTclust15dfPrincurveConv=rep(NA,it)
	twoTclust15dfPrincurveConv=rep(NA,it)
	twoOverlapClust50DPrincurveConv=rep(NA,it)
	twoCloseClust50DPrincurveConv=rep(NA,it)


	naomiPrincurveDip=rep(NA,it)
	singleClust50DPrincurveDip=rep(NA,it)
	fiveConcCirclePrincurveDip=rep(NA,it)
	singlePrincurveDip=rep(NA,it)
	single3DPrincurveDip=rep(NA,it)
	single10DPrincurveDip=rep(NA,it)
	singleOutlierPrincurveDip=rep(NA,it)
	largeOutlierPrincurveDip=rep(NA,it)
	single3OutliersPrincurveDip=rep(NA,it)
	twoSepPrincurveDip=rep(NA,it)
	threeSepPrincurveDip=rep(NA,it)
	threeSep3DPrincurveDip=rep(NA,it)
	threeClosePrincurveDip=rep(NA,it)
	threeNoisyPrincurveDip=rep(NA,it)
	threeDiffDiameterPrincurveDip=rep(NA,it)
	threeDiffDensityPrincurveDip=rep(NA,it)
	twoSep10DPrincurveDip=rep(NA,it)
	fourSep10DPrincurveDip=rep(NA,it)
	tenSeparatedPrincurveDip=rep(NA,it)
	tenClosePrincurveDip=rep(NA,it)
	sevenCloseDensePrincurveDip=rep(NA,it)
	fiveClosePrincurveDip=rep(NA,it)
	sevenClosePrincurveDip=rep(NA,it)
	parallelLinesPrincurveDip=rep(NA,it)
	concCirclesPrincurveDip=rep(NA,it)
	threeConcCirclesPrincurveDip=rep(NA,it)
	oneTclust10dfPrincurveDip=rep(NA,it)
	twoTclust10dfPrincurveDip=rep(NA,it)
	oneTclust5dfPrincurveDip=rep(NA,it)
	twoTclust5dfPrincurveDip=rep(NA,it)
	oneTclust15dfPrincurveDip=rep(NA,it)
	twoTclust15dfPrincurveDip=rep(NA,it)
	twoOverlapClust50DPrincurveDip=rep(NA,it)
	twoCloseClust50DPrincurveDip=rep(NA,it)
	

	naomiPrincurveSilv=rep(NA,it)
	singleClust50DPrincurveSilv=rep(NA,it)
	fiveConcCirclePrincurveSilv=rep(NA,it)
	singlePrincurveSilv=rep(NA,it)
	single3DPrincurveSilv=rep(NA,it)
	single10DPrincurveSilv=rep(NA,it)
	singleOutlierPrincurveSilv=rep(NA,it)
	largeOutlierPrincurveSilv=rep(NA,it)
	single3OutliersPrincurveSilv=rep(NA,it)
	twoSepPrincurveSilv=rep(NA,it)
	threeSepPrincurveSilv=rep(NA,it)
	threeSep3DPrincurveSilv=rep(NA,it)
	threeClosePrincurveSilv=rep(NA,it)
	threeNoisyPrincurveSilv=rep(NA,it)
	threeDiffDiameterPrincurveSilv=rep(NA,it)
	threeDiffDensityPrincurveSilv=rep(NA,it)
	twoSep10DPrincurveSilv=rep(NA,it)
	fourSep10DPrincurveSilv=rep(NA,it)
	tenSeparatedPrincurveSilv=rep(NA,it)
	tenClosePrincurveSilv=rep(NA,it)
	sevenCloseDensePrincurveSilv=rep(NA,it)
	fiveClosePrincurveSilv=rep(NA,it)
	sevenClosePrincurveSilv=rep(NA,it)
	parallelLinesPrincurveSilv=rep(NA,it)
	concCirclesPrincurveSilv=rep(NA,it)
	threeConcCirclesPrincurveSilv=rep(NA,it)
	oneTclust10dfPrincurveSilv=rep(NA,it)
	twoTclust10dfPrincurveSilv=rep(NA,it)
	oneTclust5dfPrincurveSilv=rep(NA,it)
	twoTclust5dfPrincurveSilv=rep(NA,it)
	oneTclust15dfPrincurveSilv=rep(NA,it)
	twoTclust15dfPrincurveSilv=rep(NA,it)
	twoOverlapClust50DPrincurveSilv=rep(NA,it)
	twoCloseClust50DPrincurveSilv=rep(NA,it)
#
#
#
	

	naomiSilvClock=0
	naomiDipClock=0
	naomiHopClock=0
	naomiClassicDipClock=0
	naomiClassicSilvClock=0
	naomiPCASilvClock=0
	naomiPCADipClock=0

	singleSilvClock=0
	singleDipClock=0
	singleHopClock=0
	singleClassicDipClock=0
	singleClassicSilvClock=0
	singlePCASilvClock=0
	singlePCADipClock=0
	single3DSilvClock=0
	single3DDipClock=0
	single3DHopClock=0
	single3DClassicSilvClock=0
	single3DClassicDipClock=0
	single3DPCASilvClock=0
	single3DPCADipClock=0
	single10DSilvClock=0
	single10DDipClock=0
	single10DHopClock=0
	single10DClassicSilvClock=0
	single10DClassicDipClock=0
	single10DPCASilvClock=0
	single10DPCADipClock=0
	singleOutlierSilvClock=0
	singleOutlierDipClock=0
	singleOutlierHopClock=0
	singleOutlierClassicSilvClock=0
	singleOutlierClassicDipClock=0
	singleOutlierPCASilvClock=0
	singleOutlierPCADipClock=0
	largeOutlierSilvClock=0
	largeOutlierDipClock=0
	largeOutlierHopClock=0
	largeOutlierClassicSilvClock=0
	largeOutlierClassicDipClock=0
	largeOutlierPCASilvClock=0
	largeOutlierPCADipClock=0
	single3OutliersSilvClock=0
	single3OutliersDipClock=0
	single3OutliersHopClock=0
	single3OutliersClassicSilvClock=0
	single3OutliersClassicDipClock=0
	single3OutliersPCASilvClock=0
	single3OutliersPCADipClock=0
	twoSepSilvClock=0
	twoSepDipClock=0
	twoSepHopClock=0
	twoSepClassicSilvClock=0
	twoSepClassicDipClock=0
	twoSepPCASilvClock=0
	twoSepPCADipClock=0
	threeSepSilvClock=0
	threeSepDipClock=0
	threeSepHopClock=0
	threeSepClassicSilvClock=0
	threeSepClassicDipClock=0
	threeSepPCASilvClock=0
	threeSepPCADipClock=0
	threeSep3DSilvClock=0
	threeSep3DDipClock=0
	threeSep3DHopClock=0
	threeSep3DClassicSilvClock=0
	threeSep3DClassicDipClock=0
	threeSep3DPCASilvClock=0
	threeSep3DPCADipClock=0
	threeCloseSilvClock=0
	threeCloseDipClock=0
	threeCloseHopClock=0
	threeCloseClassicSilvClock=0
	threeCloseClassicDipClock=0
	threeClosePCASilvClock=0
	threeClosePCADipClock=0
	threeNoisySilvClock=0
	threeNoisyDipClock=0
	threeNoisyHopClock=0
	threeNoisyClassicSilvClock=0
	threeNoisyClassicDipClock=0
	threeNoisyPCASilvClock=0
	threeNoisyPCADipClock=0
	threeDiffDiameterSilvClock=0
	threeDiffDiameterDipClock=0
	threeDiffDiameterHopClock=0
	threeDiffDiameterClassicSilvClock=0
	threeDiffDiameterClassicDipClock=0
	threeDiffDiameterPCASilvClock=0
	threeDiffDiameterPCADipClock=0
	threeDiffDensitySilvClock=0
	threeDiffDensityDipClock=0
	threeDiffDensityHopClock=0
	threeDiffDensityClassicSilvClock=0
	threeDiffDensityClassicDipClock=0
	threeDiffDensityPCASilvClock=0
	threeDiffDensityPCADipClock=0
	twoSep10DSilvClock=0
	twoSep10DDipClock=0
	twoSep10DHopClock=0
	twoSep10DClassicSilvClock=0
	twoSep10DClassicDipClock=0
	twoSep10DPCASilvClock=0
	twoSep10DPCADipClock=0
	fourSep10DSilvClock=0
	fourSep10DDipClock=0
	fourSep10DHopClock=0
	fourSep10DClassicSilvClock=0
	fourSep10DClassicDipClock=0
	fourSep10DPCASilvClock=0
	fourSep10DPCADipClock=0
	tenSeparatedSilvClock=0
	tenSeparatedDipClock=0
	tenSeparatedHopClock=0
	tenSeparatedClassicSilvClock=0
	tenSeparatedClassicDipClock=0
	tenSeparatedPCASilvClock=0
	tenSeparatedPCADipClock=0
	tenCloseSilvClock=0
	tenCloseDipClock=0
	tenCloseHopClock=0
	tenCloseClassicSilvClock=0
	tenCloseClassicDipClock=0
	tenClosePCASilvClock=0
	tenClosePCADipClock=0
	fiveCloseSilvClock=0
	fiveCloseDipClock=0
	fiveCloseHopClock=0
	fiveCloseClassicSilvClock=0
	fiveCloseClassicDipClock=0
	fiveClosePCASilvClock=0
	fiveClosePCADipClock=0
	sevenCloseSilvClock=0
	sevenCloseDipClock=0
	sevenCloseHopClock=0
	sevenCloseClassicSilvClock=0
	sevenCloseClassicDipClock=0
	sevenClosePCASilvClock=0
	sevenClosePCADipClock=0
	sevenCloseDenseSilvClock=0
	sevenCloseDenseDipClock=0
	sevenCloseDenseHopClock=0
	sevenCloseDenseClassicSilvClock=0
	sevenCloseDenseClassicDipClock=0
	sevenCloseDensePCASilvClock=0
	sevenCloseDensePCADipClock=0
	parallelLinesSilvClock=0
	parallelLinesDipClock=0
	parallelLinesHopClock=0
	parallelLinesClassicSilvClock=0
	parallelLinesClassicDipClock=0
	parallelLinesPCASilvClock=0
	parallelLinesPCADipClock=0
	concCirclesSilvClock=0
	concCirclesDipClock=0
	concCirclesHopClock=0
	concCirclesClassicSilvClock=0
	concCirclesClassicDipClock=0
	concCirclesPCASilvClock=0
	concCirclesPCADipClock=0
	threeConcCirclesSilvClock=0
	threeConcCirclesDipClock=0
	threeConcCirclesHopClock=0
	threeConcCirclesClassicSilvClock=0
	threeConcCirclesClassicDipClock=0
	threeConcCirclesPCASilvClock=0
	threeConcCirclesPCADipClock=0
	oneTclust5dfSilvClock=0
	oneTclust5dfDipClock=0
	oneTclust5dfHopClock=0
	oneTclust5dfClassicSilvClock=0
	oneTclust5dfClassicDipClock=0
	oneTclust5dfPCASilvClock=0
	oneTclust5dfPCADipClock=0
	twoTclust5dfSilvClock=0
	twoTclust5dfDipClock=0
	twoTclust5dfHopClock=0
	twoTclust5dfClassicSilvClock=0
	twoTclust5dfClassicDipClock=0
	twoTclust5dfPCASilvClock=0
	twoTclust5dfPCADipClock=0
	oneTclust10dfSilvClock=0
	oneTclust10dfDipClock=0
	oneTclust10dfHopClock=0
	oneTclust10dfClassicSilvClock=0
	oneTclust10dfClassicDipClock=0
	oneTclust10dfPCASilvClock=0
	oneTclust10dfPCADipClock=0
	twoTclust10dfSilvClock=0
	twoTclust10dfDipClock=0
	twoTclust10dfHopClock=0
	twoTclust10dfClassicSilvClock=0
	twoTclust10dfClassicDipClock=0
	twoTclust10dfPCASilvClock=0
	twoTclust10dfPCADipClock=0
	oneTclust15dfSilvClock=0
	oneTclust15dfDipClock=0
	oneTclust15dfHopClock=0
	oneTclust15dfClassicSilvClock=0
	oneTclust15dfClassicDipClock=0
	oneTclust15dfPCASilvClock=0
	oneTclust15dfPCADipClock=0
	twoTclust15dfSilvClock=0
	twoTclust15dfDipClock=0
	twoTclust15dfHopClock=0
	twoTclust15dfClassicSilvClock=0
	twoTclust15dfClassicDipClock=0
	twoTclust15dfPCASilvClock=0
	twoTclust15dfPCADipClock=0
	singleClust50DSilvClock=0
	singleClust50DDipClock=0
	singleClust50DHopClock=0
	singleClust50DClassicDipClock=0
	singleClust50DClassicSilvClock=0
	singleClust50DPCASilvClock=0
	singleClust50DPCADipClock=0
	fiveConcCircleSilvClock=0
	fiveConcCircleDipClock=0
	fiveConcCircleHopClock=0
	fiveConcCircleClassicDipClock=0
	fiveConcCircleClassicSilvClock=0
	fiveConcCirclePCASilvClock=0
	fiveConcCirclePCADipClock=0
	twoCloseClust50DSilvClock=0
	twoCloseClust50DDipClock=0
	twoCloseClust50DHopClock=0
	twoCloseClust50DClassicDipClock=0
	twoCloseClust50DClassicSilvClock=0
	twoCloseClust50DPCASilvClock=0
	twoCloseClust50DPCADipClock=0
	twoOverlapClust50DSilvClock=0
	twoOverlapClust50DDipClock=0
	twoOverlapClust50DHopClock=0
	twoOverlapClust50DClassicDipClock=0
	twoOverlapClust50DClassicSilvClock=0
	twoOverlapClust50DPCASilvClock=0
	twoOverlapClust50DPCADipClock=0

	singleClust50DPrincurveDipClock=0
	fiveConcCirclePrincurveDipClock=0
	naomiPrincurveDipClock=0
	naomiPrincurveSilvClock=0
	singlePrincurveDipClock=0
	single3DPrincurveDipClock=0
	single10DPrincurveDipClock=0
	singleOutlierPrincurveDipClock=0
	largeOutlierPrincurveDipClock=0
	single3OutliersPrincurveDipClock=0
	twoSepPrincurveDipClock=0
	threeSepPrincurveDipClock=0
	threeSep3DPrincurveDipClock=0
	threeClosePrincurveDipClock=0
	threeNoisyPrincurveDipClock=0
	threeDiffDiameterPrincurveDipClock=0
	threeDiffDensityPrincurveDipClock=0
	twoSep10DPrincurveDipClock=0
	fourSep10DPrincurveDipClock=0
	tenSeparatedPrincurveDipClock=0
	tenClosePrincurveDipClock=0
	sevenCloseDensePrincurveDipClock=0
	fiveClosePrincurveDipClock=0
	sevenClosePrincurveDipClock=0
	parallelLinesPrincurveDipClock=0
	concCirclesPrincurveDipClock=0
	threeConcCirclesPrincurveDipClock=0
	oneTclust10dfPrincurveDipClock=0
	twoTclust10dfPrincurveDipClock=0
	oneTclust5dfPrincurveDipClock=0
	twoTclust5dfPrincurveDipClock=0
	oneTclust15dfPrincurveDipClock=0
	twoTclust15dfPrincurveDipClock=0
	twoOverlapClust50DPrincurveDipClock=0
	twoCloseClust50DPrincurveDipClock=0
	singleClust50DPrincurveSilvClock=0
	fiveConcCirclePrincurveSilvClock=0
	singlePrincurveSilvClock=0
	single3DPrincurveSilvClock=0
	single10DPrincurveSilvClock=0
	singleOutlierPrincurveSilvClock=0
	largeOutlierPrincurveSilvClock=0
	single3OutliersPrincurveSilvClock=0
	twoSepPrincurveSilvClock=0
	threeSepPrincurveSilvClock=0
	threeSep3DPrincurveSilvClock=0
	threeClosePrincurveSilvClock=0
	threeNoisyPrincurveSilvClock=0
	threeDiffDiameterPrincurveSilvClock=0
	threeDiffDensityPrincurveSilvClock=0
	twoSep10DPrincurveSilvClock=0
	fourSep10DPrincurveSilvClock=0
	tenSeparatedPrincurveSilvClock=0
	tenClosePrincurveSilvClock=0
	sevenCloseDensePrincurveSilvClock=0
	fiveClosePrincurveSilvClock=0
	sevenClosePrincurveSilvClock=0
	parallelLinesPrincurveSilvClock=0
	concCirclesPrincurveSilvClock=0
	threeConcCirclesPrincurveSilvClock=0
	oneTclust10dfPrincurveSilvClock=0
	twoTclust10dfPrincurveSilvClock=0
	oneTclust5dfPrincurveSilvClock=0
	twoTclust5dfPrincurveSilvClock=0
	oneTclust15dfPrincurveSilvClock=0
	twoTclust15dfPrincurveSilvClock=0
	twoOverlapClust50DPrincurveSilvClock=0
	twoCloseClust50DPrincurveSilvClock=0

#
#
#



	for(i in 1:it){
		loopTimer <- proc.time()		#starts loop stopwatch
	
	#create
		naomiClust<-naomicustomcluster(50,2)

		singleClust<-cluster(50,100,100,2)
		singleClust3D<-cluster3D(50,100,100,100,2)
		singleClust10D<-cluster10D(50,2,2)
		singleClust50D<-cluster50D(100,2,2)
		singleClustOutlier<-clusterWithOutlier()
		largeClustOutlier<-largeClusterWithOutlier()
		singleClust3Outliers<-clusterWith3Outliers()
		twoSepClust<-twoSeparatedClusters()
		threeSepClust<-threeSeparatedClusters()
		threeSepClust3D<-threeSeparatedClusters3D()
		threeCloseClust<-threeCloseClusters()
		threeNoisyClust<-threeNoisyClusters()
		threeClustDiffDiameter<-threeClustersVaryingDiameter()
		threeClustDiffDensity<-threeClustersVaryingDensity()
		twoSepClust10D<-twoSeparatedClusters10D()	
		fourSepClust10D<-fourSeparatedClusters10D()
		tenSeparatedClusters<-tenSeparatedClusters()
		tenCloseClusters<-tenCloseClusters()
		sevenCloseDenseClusters<-sevenCloseDenseClusters()		
		fiveCloseClusters<-fiveCloseClusters()
		sevenCloseClusters<-sevenCloseClusters()
		twoSepLines<-parallelLines(100,25,20)
		concCircles<-concentricCircles(100)
		threeConcCircles<-threeConcentricCircles(100)
		fiveConcCircles<-fiveConcentricCircles(100)
		oneTclust10df<-Tclust(100,100,100,10)
		oneTclust5df<-Tclust(100,100,100,5)
		oneTclust15df<-Tclust(100,100,100,15)
		twoTclust10df<-twoTclust(100,10)
		twoTclust5df<-twoTclust(100,5)
		twoTclust15df<-twoTclust(100,15)
		twoOverlapClust50D<-twoClusters50D(3)
		twoCloseClust50D<-twoClusters50D(5)

	#dist
		checkpoint<-Sys.time()
		naomiSilv[i]<-silverman.test(dist(naomiClust),1,adjust=TRUE)@p_value
		naomiSilvClock = naomiSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleSilv[i]<-silverman.test(dist(singleClust),1,adjust=TRUE)@p_value
		singleSilvClock = singleSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DSilv[i]<-silverman.test(dist(singleClust3D),1,adjust=TRUE)@p_value
		single3DSilvClock = single3DSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DSilv[i]<-silverman.test(dist(singleClust10D),1,adjust=TRUE)@p_value
		single10DSilvClock = single10DSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierSilv[i]<-silverman.test(dist(singleClustOutlier),1,adjust=TRUE)@p_value
		singleOutlierSilvClock = singleOutlierSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierSilv[i]<-silverman.test(dist(largeClustOutlier),1,adjust=TRUE)@p_value
		largeOutlierSilvClock = largeOutlierSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersSilv[i]<-silverman.test(dist(singleClust3Outliers),1,adjust=TRUE)@p_value
		single3OutliersSilvClock = single3OutliersSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepSilv[i]<-silverman.test(dist(twoSepClust),1,adjust=TRUE)@p_value
		twoSepSilvClock = twoSepSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepSilv[i]<-silverman.test(dist(threeSepClust),1,adjust=TRUE)@p_value
		threeSepSilvClock = threeSepSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DSilv[i]<-silverman.test(dist(threeSepClust3D),1,adjust=TRUE)@p_value
		threeSep3DSilvClock = threeSep3DSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeCloseSilv[i]<-silverman.test(dist(threeCloseClust),1,adjust=TRUE)@p_value
		threeCloseSilvClock = threeCloseSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisySilv[i]<-silverman.test(dist(threeNoisyClust),1,adjust=TRUE)@p_value
		threeNoisySilvClock = threeNoisySilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterSilv[i]<-silverman.test(dist(threeClustDiffDiameter),1,adjust=TRUE)@p_value
		threeDiffDiameterSilvClock = threeDiffDiameterSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensitySilv[i]<-silverman.test(dist(threeClustDiffDensity),1,adjust=TRUE)@p_value
		threeDiffDensitySilvClock = threeDiffDensitySilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DSilv[i]<-silverman.test(dist(twoSepClust10D),1,adjust=TRUE)@p_value
		twoSep10DSilvClock = twoSep10DSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DSilv[i]<-silverman.test(dist(fourSepClust10D),1,adjust=TRUE)@p_value
		fourSep10DSilvClock = fourSep10DSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedSilv[i]<-silverman.test(dist(tenSeparatedClusters),1,adjust=TRUE)@p_value
		tenSeparatedSilvClock = tenSeparatedSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenCloseSilv[i]<-silverman.test(dist(tenCloseClusters),1,adjust=TRUE)@p_value
		tenCloseSilvClock = tenCloseSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDenseSilv[i]<-silverman.test(dist(sevenCloseDenseClusters),1,adjust=TRUE)@p_value
		sevenCloseDenseSilvClock = sevenCloseDenseSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveCloseSilv[i]<-silverman.test(dist(fiveCloseClusters),1,adjust=TRUE)@p_value
		fiveCloseSilvClock = fiveCloseSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseSilv[i]<-silverman.test(dist(sevenCloseClusters),1,adjust=TRUE)@p_value
		sevenCloseSilvClock = sevenCloseSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesSilv[i]<-silverman.test(dist(twoSepLines),1,adjust=TRUE)@p_value
		parallelLinesSilvClock = parallelLinesSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesSilv[i]<-silverman.test(dist(concCircles),1,adjust=TRUE)@p_value
		concCirclesSilvClock = concCirclesSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesSilv[i]<-silverman.test(dist(threeConcCircles),1,adjust=TRUE)@p_value
		threeConcCirclesSilvClock = threeConcCirclesSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfSilv[i]<-silverman.test(dist(oneTclust10df),1,adjust=TRUE)@p_value
		oneTclust10dfSilvClock = oneTclust10dfSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfSilv[i]<-silverman.test(dist(twoTclust10df),1,adjust=TRUE)@p_value
		twoTclust10dfSilvClock = twoTclust10dfSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfSilv[i]<-silverman.test(dist(oneTclust5df),1,adjust=TRUE)@p_value
		oneTclust5dfSilvClock = oneTclust5dfSilvClock + as.numeric(Sys.time()-checkpoint)
		
		checkpoint<-Sys.time()
		twoTclust5dfSilv[i]<-silverman.test(dist(twoTclust5df),1,adjust=TRUE)@p_value
		twoTclust5dfSilvClock = twoTclust5dfSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfSilv[i]<-silverman.test(dist(oneTclust15df),1,adjust=TRUE)@p_value
		oneTclust15dfSilvClock = oneTclust15dfSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfSilv[i]<-silverman.test(dist(twoTclust15df),1,adjust=TRUE)@p_value
		twoTclust15dfSilvClock = twoTclust15dfSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DSilv[i]<-silverman.test(dist(singleClust50D),1,adjust=TRUE)@p_value
		singleClust50DSilvClock = singleClust50DSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCircleSilv[i]<-silverman.test(dist(fiveConcCircles),1,adjust=TRUE)@p_value
		fiveConcCircleSilvClock = fiveConcCircleSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoOverlapClust50DSilv[i]<-silverman.test(dist(twoOverlapClust50D),1,adjust=TRUE)@p_value
		twoOverlapClust50DSilvClock = twoOverlapClust50DSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DSilv[i]<-silverman.test(dist(twoCloseClust50D),1,adjust=TRUE)@p_value
		twoCloseClust50DSilvClock = twoCloseClust50DSilvClock + as.numeric(Sys.time()-checkpoint)


#
		checkpoint<-Sys.time()
		naomiDip[i]<-dip.test(dist(naomiClust))$p.value
		naomiDipClock = naomiDipClock + as.numeric(Sys.time()-checkpoint)


		checkpoint<-Sys.time()
		twoOverlapClust50DDip[i]<-dip.test(dist(twoOverlapClust50D))$p.value
		twoOverlapClust50DDipClock = twoOverlapClust50DDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DDip[i]<-dip.test(dist(twoCloseClust50D))$p.value		
		twoCloseClust50DDipClock = twoCloseClust50DDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DDip[i]<-dip.test(dist(singleClust50D))$p.value
		singleClust50DDipClock = singleClust50DDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCircleDip[i]<-dip.test(dist(fiveConcCircles))$p.value
		fiveConcCircleDipClock = fiveConcCircleDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleDip[i]<-dip.test(dist(singleClust))$p.value
		singleDipClock = singleDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DDip[i]<-dip.test(dist(singleClust3D))$p.value
		single3DDipClock = single3DDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DDip[i]<-dip.test(dist(singleClust10D))$p.value
		single10DDipClock = single10DDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierDip[i]<-dip.test(dist(singleClustOutlier))$p.value
		singleOutlierDipClock = singleOutlierDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierDip[i]<-dip.test(dist(largeClustOutlier))$p.value
		largeOutlierDipClock = largeOutlierDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersDip[i]<-dip.test(dist(singleClust3Outliers))$p.value
		single3OutliersDipClock = single3OutliersDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepDip[i]<-dip.test(dist(twoSepClust))$p.value
		twoSepDipClock = twoSepDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeCloseDip[i]<-dip.test(dist(threeCloseClust))$p.value
		threeCloseDipClock = threeCloseDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepDip[i]<-dip.test(dist(threeSepClust))$p.value
		threeSepDipClock = threeSepDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DDip[i]<-dip.test(dist(threeSepClust3D))$p.value
		threeSep3DDipClock = threeSep3DDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyDip[i]<-dip.test(dist(threeNoisyClust))$p.value
		threeNoisyDipClock = threeNoisyDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterDip[i]<-dip.test(dist(threeClustDiffDiameter))$p.value
		threeDiffDiameterDipClock = threeDiffDiameterDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityDip[i]<-dip.test(dist(threeClustDiffDensity))$p.value
		threeDiffDensityDipClock = threeDiffDensityDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DDip[i]<-dip.test(dist(twoSepClust10D))$p.value
		twoSep10DDipClock = twoSep10DDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DDip[i]<-dip.test(dist(fourSepClust10D))$p.value
		fourSep10DDipClock = fourSep10DDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedDip[i]<-dip.test(dist(tenSeparatedClusters))$p.value
		tenSeparatedDipClock = tenSeparatedDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenCloseDip[i]<-dip.test(dist(tenCloseClusters))$p.value
		tenCloseDipClock = tenCloseDipClock + as.numeric(Sys.time()-checkpoint)


		checkpoint<-Sys.time()
		sevenCloseDenseDip[i]<-dip.test(dist(sevenCloseDenseClusters))$p.value		
		sevenCloseDenseDipClock = sevenCloseDenseDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveCloseDip[i]<-dip.test(dist(fiveCloseClusters))$p.value
		fiveCloseDipClock = fiveCloseDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDip[i]<-dip.test(dist(sevenCloseClusters))$p.value
		sevenCloseDipClock = sevenCloseDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesDip[i]<-dip.test(dist(twoSepLines))$p.value
		parallelLinesDipClock = parallelLinesDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesDip[i]<-dip.test(dist(concCircles))$p.value
		concCirclesDipClock = concCirclesDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesDip[i]<-dip.test(dist(threeConcCircles))$p.value
		threeConcCirclesDipClock = threeConcCirclesDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfDip[i]<-dip.test(dist(oneTclust10df))$p.value
		oneTclust10dfDipClock = oneTclust10dfDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfDip[i]<-dip.test(dist(twoTclust10df))$p.value
		twoTclust10dfDipClock = twoTclust10dfDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfDip[i]<-dip.test(dist(oneTclust5df))$p.value
		oneTclust5dfDipClock = oneTclust5dfDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfDip[i]<-dip.test(dist(twoTclust5df))$p.value
		twoTclust5dfDipClock = twoTclust5dfDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfDip[i]<-dip.test(dist(oneTclust15df))$p.value
		oneTclust15dfDipClock = oneTclust15dfDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfDip[i]<-dip.test(dist(twoTclust15df))$p.value
		twoTclust15dfDipClock = twoTclust15dfDipClock + as.numeric(Sys.time()-checkpoint)


		

	#Hop
		checkpoint<-Sys.time()
		naomiHop[i]<-hopkins(as.data.frame(naomiClust), n = nrow(naomiClust)/10)$H
		naomiHopClock = naomiHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleHop[i]<-hopkins(as.data.frame(singleClust), n = nrow(singleClust)/10)$H
		singleHopClock = singleHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DHop[i]<-hopkins(as.data.frame(singleClust3D), n = nrow(singleClust3D)/10)$H
		single3DHopClock = single3DHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DHop[i]<-hopkins(as.data.frame(singleClust10D), n = nrow(singleClust10D)/10)$H
		single10DHopClock = single10DHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierHop[i]<-hopkins(as.data.frame(singleClustOutlier), n = nrow(singleClustOutlier)/10)$H
		singleOutlierHopClock = singleOutlierHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierHop[i]<-hopkins(as.data.frame(largeClustOutlier), n = nrow(largeClustOutlier)/10)$H
		largeOutlierHopClock = largeOutlierHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersHop[i]<-hopkins(as.data.frame(singleClust3Outliers), n = nrow(singleClust3Outliers)/10)$H
		single3OutliersHopClock = single3OutliersHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepHop[i]<-hopkins(as.data.frame(twoSepClust), n = nrow(twoSepClust)/10)$H
		twoSepHopClock = twoSepHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepHop[i]<-hopkins(as.data.frame(threeSepClust), n = nrow(threeSepClust)/10)$H
		threeSepHopClock = threeSepHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DHop[i]<-hopkins(as.data.frame(threeSepClust3D), n = nrow(threeSepClust3D)/10)$H
		threeSep3DHopClock = threeSep3DHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeCloseHop[i]<-hopkins(as.data.frame(threeCloseClust), n = nrow(threeCloseClust)/10)$H
		threeCloseHopClock = threeCloseHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyHop[i]<-hopkins(as.data.frame(threeNoisyClust), n = nrow(threeNoisyClust)/10)$H
		threeNoisyHopClock = threeNoisyHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterHop[i]<-hopkins(as.data.frame(threeClustDiffDiameter), n = nrow(threeClustDiffDiameter)/10)$H
		threeDiffDiameterHopClock = threeDiffDiameterHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityHop[i]<-hopkins(as.data.frame(threeClustDiffDensity), n = nrow(threeClustDiffDensity)/10)$H
		threeDiffDensityHopClock = threeDiffDensityHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DHop[i]<-hopkins(as.data.frame(twoSepClust10D), n = nrow(twoSepClust10D)/10)$H
		twoSep10DHopClock = twoSep10DHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DHop[i]<-hopkins(as.data.frame(fourSepClust10D), n = nrow(fourSepClust10D)/10)$H
		fourSep10DHopClock = fourSep10DHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedHop[i]<-hopkins(as.data.frame(tenSeparatedClusters), n = nrow(tenSeparatedClusters)/10)$H
		tenSeparatedHopClock = tenSeparatedHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenCloseHop[i]<-hopkins(as.data.frame(tenCloseClusters), n = nrow(tenCloseClusters)/10)$H
		tenCloseHopClock = tenCloseHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDenseHop[i]<-hopkins(as.data.frame(sevenCloseDenseClusters), n = nrow(sevenCloseDenseClusters)/10)$H		
		sevenCloseDenseHopClock = sevenCloseDenseHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveCloseHop[i]<-hopkins(as.data.frame(fiveCloseClusters), n = nrow(fiveCloseClusters)/10)$H
		fiveCloseHopClock = fiveCloseHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseHop[i]<-hopkins(as.data.frame(sevenCloseClusters), n = nrow(sevenCloseClusters)/10)$H
		sevenCloseHopClock = sevenCloseHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesHop[i]<-hopkins(as.data.frame(twoSepLines), n = nrow(twoSepLines)/10)$H
		parallelLinesHopClock = parallelLinesHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesHop[i]<-hopkins(as.data.frame(concCircles), n = nrow(concCircles)/10)$H
		concCirclesHopClock = concCirclesHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesHop[i]<-hopkins(as.data.frame(threeConcCircles), n = nrow(threeConcCircles)/10)$H
		threeConcCirclesHopClock = threeConcCirclesHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfHop[i]<-hopkins(as.data.frame(oneTclust10df), n = nrow(oneTclust10df)/10)$H
		oneTclust10dfHopClock = oneTclust10dfHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfHop[i]<-hopkins(as.data.frame(twoTclust10df), n = nrow(twoTclust10df)/10)$H
		twoTclust10dfHopClock = twoTclust10dfHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfHop[i]<-hopkins(as.data.frame(oneTclust5df), n = nrow(oneTclust5df)/10)$H
		oneTclust5dfHopClock = oneTclust5dfHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfHop[i]<-hopkins(as.data.frame(twoTclust5df), n = nrow(twoTclust5df)/10)$H
		twoTclust5dfHopClock = twoTclust5dfHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfHop[i]<-hopkins(as.data.frame(oneTclust15df), n = nrow(oneTclust15df)/10)$H
		oneTclust15dfHopClock = oneTclust15dfHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfHop[i]<-hopkins(as.data.frame(twoTclust15df), n = nrow(twoTclust15df)/10)$H
		twoTclust15dfHopClock = twoTclust15dfHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DHop[i]<-hopkins(as.data.frame(singleClust50D), n = nrow(singleClust50D)/10)$H
		singleClust50DHopClock = singleClust50DHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCircleHop[i]<-hopkins(as.data.frame(fiveConcCircles), n = nrow(fiveConcCircles)/10)$H	
		fiveConcCircleHopClock = fiveConcCircleHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoOverlapClust50DHop[i]<-hopkins(as.data.frame(twoOverlapClust50D), n = nrow(twoOverlapClust50D)/10)$H
		twoOverlapClust50DHopClock = twoOverlapClust50DHopClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DHop[i]<-hopkins(as.data.frame(twoCloseClust50D), n = nrow(twoCloseClust50D)/10)$H
		twoCloseClust50DHopClock = twoCloseClust50DHopClock + as.numeric(Sys.time()-checkpoint)


	#Classic
		checkpoint<-Sys.time()
		singleClassicDip[i]<-dip.test(singleClust)$p.value
		singleClassicDipClock = singleClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DClassicDip[i]<-dip.test(singleClust3D)$p.value
		single3DClassicDipClock = single3DClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DClassicDip[i]<-dip.test(singleClust10D)$p.value
		single10DClassicDipClock = single10DClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierClassicDip[i]<-dip.test(singleClustOutlier)$p.value
		singleOutlierClassicDipClock = singleOutlierClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierClassicDip[i]<-dip.test(largeClustOutlier)$p.value
		largeOutlierClassicDipClock = largeOutlierClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersClassicDip[i]<-dip.test(singleClust3Outliers)$p.value
		single3OutliersClassicDipClock = single3OutliersClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepClassicDip[i]<-dip.test(twoSepClust)$p.value
		twoSepClassicDipClock = twoSepClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepClassicDip[i]<-dip.test(threeSepClust)$p.value
		threeSepClassicDipClock = threeSepClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DClassicDip[i]<-dip.test(threeSepClust3D)$p.value
		threeSep3DClassicDipClock = threeSep3DClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeCloseClassicDip[i]<-dip.test(threeCloseClust)$p.value
		threeCloseClassicDipClock = threeCloseClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyClassicDip[i]<-dip.test(threeNoisyClust)$p.value
		threeNoisyClassicDipClock = threeNoisyClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterClassicDip[i]<-dip.test(threeClustDiffDiameter)$p.value
		threeDiffDiameterClassicDipClock = threeDiffDiameterClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityClassicDip[i]<-dip.test(threeClustDiffDensity)$p.value
		threeDiffDensityClassicDipClock = threeDiffDensityClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DClassicDip[i]<-dip.test(twoSepClust10D)$p.value
		twoSep10DClassicDipClock = twoSep10DClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DClassicDip[i]<-dip.test(fourSepClust10D)$p.value
		fourSep10DClassicDipClock = fourSep10DClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedClassicDip[i]<-dip.test(tenSeparatedClusters)$p.value
		tenSeparatedClassicDipClock = tenSeparatedClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenCloseClassicDip[i]<-dip.test(tenCloseClusters)$p.value
		tenCloseClassicDipClock = tenCloseClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDenseClassicDip[i]<-dip.test(sevenCloseDenseClusters)$p.value
		sevenCloseDenseClassicDipClock = sevenCloseDenseClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveCloseClassicDip[i]<-dip.test(fiveCloseClusters)$p.value
		fiveCloseClassicDipClock = fiveCloseClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseClassicDip[i]<-dip.test(sevenCloseClusters)$p.value
		sevenCloseClassicDipClock = sevenCloseClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesClassicDip[i]<-dip.test(twoSepLines)$p.value
		parallelLinesClassicDipClock = parallelLinesClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesClassicDip[i]<-dip.test(concCircles)$p.value
		concCirclesClassicDipClock = concCirclesClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesClassicDip[i]<-dip.test(threeConcCircles)$p.value
		threeConcCirclesClassicDipClock = threeConcCirclesClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfClassicDip[i]<-dip.test(oneTclust10df)$p.value
		oneTclust10dfClassicDipClock = oneTclust10dfClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfClassicDip[i]<-dip.test(twoTclust10df)$p.value
		twoTclust10dfClassicDipClock = twoTclust10dfClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfClassicDip[i]<-dip.test(oneTclust5df)$p.value
		oneTclust5dfClassicDipClock = oneTclust5dfClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfClassicDip[i]<-dip.test(twoTclust5df)$p.value
		twoTclust5dfClassicDipClock = twoTclust5dfClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfClassicDip[i]<-dip.test(oneTclust15df)$p.value
		oneTclust15dfClassicDipClock = oneTclust15dfClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfClassicDip[i]<-dip.test(twoTclust15df)$p.value
		twoTclust15dfClassicDipClock = twoTclust15dfClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DClassicDip[i]<-dip.test(singleClust50D)$p.value
		singleClust50DClassicDipClock = singleClust50DClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCircleClassicDip[i]<-dip.test(fiveConcCircles)$p.value
		fiveConcCircleClassicDipClock = fiveConcCircleClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoOverlapClust50DClassicDip[i]<-dip.test(twoOverlapClust50D)$p.value		
		twoOverlapClust50DClassicDipClock = twoOverlapClust50DClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DClassicDip[i]<-dip.test(twoCloseClust50D)$p.value
		twoCloseClust50DClassicDipClock = twoCloseClust50DClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		naomiClassicDip[i]<-dip.test(naomiClust)$p.value
		naomiClassicDipClock = naomiClassicDipClock + as.numeric(Sys.time()-checkpoint)
#

		checkpoint<-Sys.time()
		naomiClassicSilv[i]<-silverman.test(naomiClust,1,adjust=TRUE)@p_value
		naomiClassicSilvClock = naomiClassicSilvClock + as.numeric(Sys.time()-checkpoint)


		checkpoint<-Sys.time()
		twoOverlapClust50DClassicSilv[i]<-silverman.test(twoOverlapClust50D,1,adjust=TRUE)@p_value
		twoOverlapClust50DClassicSilvClock = twoOverlapClust50DClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DClassicSilv[i]<-silverman.test(twoCloseClust50D,1,adjust=TRUE)@p_value
		twoCloseClust50DClassicSilvClock = twoCloseClust50DClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DClassicSilv[i]<-silverman.test(singleClust50D,1,adjust=TRUE)@p_value
		singleClust50DClassicSilvClock = singleClust50DClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCircleClassicSilv[i]<-silverman.test(fiveConcCircles,1,adjust=TRUE)@p_value
		fiveConcCircleClassicSilvClock = fiveConcCircleClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClassicSilv[i]<-silverman.test(singleClust,1,adjust=TRUE)@p_value
		singleClassicSilvClock = singleClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DClassicSilv[i]<-silverman.test(singleClust3D,1,adjust=TRUE)@p_value
		single3DClassicSilvClock = single3DClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DClassicSilv[i]<-silverman.test(singleClust10D,1,adjust=TRUE)@p_value
		single10DClassicSilvClock = single10DClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierClassicSilv[i]<-silverman.test(singleClustOutlier,1,adjust=TRUE)@p_value
		singleOutlierClassicSilvClock = singleOutlierClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierClassicSilv[i]<-silverman.test(largeClustOutlier,1,adjust=TRUE)@p_value
		largeOutlierClassicSilvClock = largeOutlierClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersClassicSilv[i]<-silverman.test(singleClust3Outliers,1,adjust=TRUE)@p_value
		single3OutliersClassicSilvClock = single3OutliersClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepClassicSilv[i]<-silverman.test(twoSepClust,1,adjust=TRUE)@p_value
		twoSepClassicSilvClock = twoSepClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepClassicSilv[i]<-silverman.test(threeSepClust,1,adjust=TRUE)@p_value
		threeSepClassicSilvClock = threeSepClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DClassicSilv[i]<-silverman.test(threeSepClust3D,1,adjust=TRUE)@p_value
		threeSep3DClassicSilvClock = threeSep3DClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeCloseClassicSilv[i]<-silverman.test(threeCloseClust,1,adjust=TRUE)@p_value
		threeCloseClassicSilvClock = threeCloseClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyClassicSilv[i]<-silverman.test(threeNoisyClust,1,adjust=TRUE)@p_value
		threeNoisyClassicSilvClock = threeNoisyClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterClassicSilv[i]<-silverman.test(threeClustDiffDiameter,1,adjust=TRUE)@p_value
		threeDiffDiameterClassicSilvClock = threeDiffDiameterClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityClassicSilv[i]<-silverman.test(threeClustDiffDensity,1,adjust=TRUE)@p_value
		threeDiffDensityClassicSilvClock = threeDiffDensityClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DClassicSilv[i]<-silverman.test(twoSepClust10D,1,adjust=TRUE)@p_value
		twoSep10DClassicSilvClock = twoSep10DClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DClassicSilv[i]<-silverman.test(fourSepClust10D,1,adjust=TRUE)@p_value
		fourSep10DClassicSilvClock = fourSep10DClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedClassicSilv[i]<-silverman.test(tenSeparatedClusters,1,adjust=TRUE)@p_value
		tenSeparatedClassicSilvClock = tenSeparatedClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenCloseClassicSilv[i]<-silverman.test(tenCloseClusters,1,adjust=TRUE)@p_value
		tenCloseClassicSilvClock = tenCloseClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDenseClassicSilv[i]<-silverman.test(sevenCloseDenseClusters,1,adjust=TRUE)@p_value
		sevenCloseDenseClassicSilvClock = sevenCloseDenseClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveCloseClassicSilv[i]<-silverman.test(fiveCloseClusters,1,adjust=TRUE)@p_value
		fiveCloseClassicSilvClock = fiveCloseClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseClassicSilv[i]<-silverman.test(sevenCloseClusters,1,adjust=TRUE)@p_value
		sevenCloseClassicSilvClock = sevenCloseClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesClassicSilv[i]<-silverman.test(twoSepLines,1,adjust=TRUE)@p_value
		parallelLinesClassicSilvClock = parallelLinesClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesClassicSilv[i]<-silverman.test(concCircles,1,adjust=TRUE)@p_value
		concCirclesClassicSilvClock = concCirclesClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesClassicSilv[i]<-silverman.test(threeConcCircles,1,adjust=TRUE)@p_value
		threeConcCirclesClassicSilvClock = threeConcCirclesClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfClassicSilv[i]<-silverman.test(oneTclust10df,1,adjust=TRUE)@p_value
		oneTclust10dfClassicSilvClock = oneTclust10dfClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfClassicSilv[i]<-silverman.test(twoTclust10df,1,adjust=TRUE)@p_value
		twoTclust10dfClassicSilvClock = twoTclust10dfClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfClassicSilv[i]<-silverman.test(oneTclust5df,1,adjust=TRUE)@p_value
		oneTclust5dfClassicSilvClock = oneTclust5dfClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfClassicSilv[i]<-silverman.test(twoTclust5df,1,adjust=TRUE)@p_value
		twoTclust5dfClassicSilvClock = twoTclust5dfClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfClassicSilv[i]<-silverman.test(oneTclust15df,1,adjust=TRUE)@p_value
		oneTclust15dfClassicSilvClock = oneTclust15dfClassicSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfClassicSilv[i]<-silverman.test(twoTclust15df,1,adjust=TRUE)@p_value
		twoTclust15dfClassicSilvClock = twoTclust15dfClassicSilvClock + as.numeric(Sys.time()-checkpoint)

			
	#PCA	
		checkpoint<-Sys.time()
		naomiPCASilv[i]<-silverman.test(prcomp(naomiClust)$x[,1],1,adjust=TRUE)@p_value
		naomiPCASilvClock = naomiPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singlePCASilv[i]<-silverman.test(prcomp(singleClust)$x[,1],1,adjust=TRUE)@p_value
		singlePCASilvClock = singlePCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DPCASilv[i]<-silverman.test(prcomp(singleClust3D)$x[,1],1,adjust=TRUE)@p_value
		single3DPCASilvClock = single3DPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DPCASilv[i]<-silverman.test(prcomp(singleClust10D)$x[,1],1,adjust=TRUE)@p_value
		single10DPCASilvClock = single10DPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierPCASilv[i]<-silverman.test(prcomp(singleClustOutlier)$x[,1],1,adjust=TRUE)@p_value
		singleOutlierPCASilvClock = singleOutlierPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierPCASilv[i]<-silverman.test(prcomp(largeClustOutlier)$x[,1],1,adjust=TRUE)@p_value
		largeOutlierPCASilvClock = largeOutlierPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersPCASilv[i]<-silverman.test(prcomp(singleClust3Outliers)$x[,1],1,adjust=TRUE)@p_value
		single3OutliersPCASilvClock = single3OutliersPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepPCASilv[i]<-silverman.test(prcomp(twoSepClust)$x[,1],1,adjust=TRUE)@p_value
		twoSepPCASilvClock = twoSepPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepPCASilv[i]<-silverman.test(prcomp(threeSepClust)$x[,1],1,adjust=TRUE)@p_value
		threeSepPCASilvClock = threeSepPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DPCASilv[i]<-silverman.test(prcomp(threeSepClust3D)$x[,1],1,adjust=TRUE)@p_value
		threeSep3DPCASilvClock = threeSep3DPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeClosePCASilv[i]<-silverman.test(prcomp(threeCloseClust)$x[,1],1,adjust=TRUE)@p_value
		threeClosePCASilvClock = threeClosePCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyPCASilv[i]<-silverman.test(prcomp(threeNoisyClust)$x[,1],1,adjust=TRUE)@p_value
		threeNoisyPCASilvClock = threeNoisyPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterPCASilv[i]<-silverman.test(prcomp(threeClustDiffDiameter)$x[,1],1,adjust=TRUE)@p_value
		threeDiffDiameterPCASilvClock = threeDiffDiameterPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityPCASilv[i]<-silverman.test(prcomp(threeClustDiffDensity)$x[,1],1,adjust=TRUE)@p_value
		threeDiffDensityPCASilvClock = threeDiffDensityPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DPCASilv[i]<-silverman.test(prcomp(twoSepClust10D)$x[,1],1,adjust=TRUE)@p_value
		twoSep10DPCASilvClock = twoSep10DPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DPCASilv[i]<-silverman.test(prcomp(fourSepClust10D)$x[,1],1,adjust=TRUE)@p_value
		fourSep10DPCASilvClock = fourSep10DPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedPCASilv[i]<-silverman.test(prcomp(tenSeparatedClusters)$x[,1],1,adjust=TRUE)@p_value
		tenSeparatedPCASilvClock = tenSeparatedPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenClosePCASilv[i]<-silverman.test(prcomp(tenCloseClusters)$x[,1],1,adjust=TRUE)@p_value
		tenClosePCASilvClock = tenClosePCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDensePCASilv[i]<-silverman.test(prcomp(sevenCloseDenseClusters)$x[,1],1,adjust=TRUE)@p_value
		sevenCloseDensePCASilvClock = sevenCloseDensePCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveClosePCASilv[i]<-silverman.test(prcomp(fiveCloseClusters)$x[,1],1,adjust=TRUE)@p_value
		fiveClosePCASilvClock = fiveClosePCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenClosePCASilv[i]<-silverman.test(prcomp(sevenCloseClusters)$x[,1],1,adjust=TRUE)@p_value
		sevenClosePCASilvClock = sevenClosePCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesPCASilv[i]<-silverman.test(prcomp(twoSepLines)$x[,1],1,adjust=TRUE)@p_value
		parallelLinesPCASilvClock = parallelLinesPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesPCASilv[i]<-silverman.test(prcomp(concCircles)$x[,1],1,adjust=TRUE)@p_value
		concCirclesPCASilvClock = concCirclesPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesPCASilv[i]<-silverman.test(prcomp(threeConcCircles)$x[,1],1,adjust=TRUE)@p_value
		threeConcCirclesPCASilvClock = threeConcCirclesPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfPCASilv[i]<-silverman.test(prcomp(oneTclust10df)$x[,1],1,adjust=TRUE)@p_value
		oneTclust10dfPCASilvClock = oneTclust10dfPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfPCASilv[i]<-silverman.test(prcomp(twoTclust10df)$x[,1],1,adjust=TRUE)@p_value
		twoTclust10dfPCASilvClock = twoTclust10dfPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfPCASilv[i]<-silverman.test(prcomp(oneTclust5df)$x[,1],1,adjust=TRUE)@p_value
		oneTclust5dfPCASilvClock = oneTclust5dfPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfPCASilv[i]<-silverman.test(prcomp(twoTclust5df)$x[,1],1,adjust=TRUE)@p_value
		twoTclust5dfPCASilvClock = twoTclust5dfPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfPCASilv[i]<-silverman.test(prcomp(oneTclust15df)$x[,1],1,adjust=TRUE)@p_value
		oneTclust15dfPCASilvClock = oneTclust15dfPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfPCASilv[i]<-silverman.test(prcomp(twoTclust15df)$x[,1],1,adjust=TRUE)@p_value
		twoTclust15dfPCASilvClock = twoTclust15dfPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DPCASilv[i]<-silverman.test(prcomp(singleClust50D)$x[,1],1,adjust=TRUE)@p_value
		singleClust50DPCASilvClock = singleClust50DPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCirclePCASilv[i]<-silverman.test(prcomp(fiveConcCircles)$x[,1],1,adjust=TRUE)@p_value
		fiveConcCirclePCASilvClock = fiveConcCirclePCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DPCASilv[i]<-silverman.test(prcomp(twoCloseClust50D)$x[,1],1,adjust=TRUE)@p_value
		twoCloseClust50DPCASilvClock = twoCloseClust50DPCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoOverlapClust50DPCASilv[i]<-silverman.test(prcomp(twoOverlapClust50D)$x[,1],1,adjust=TRUE)@p_value
		twoOverlapClust50DPCASilvClock = twoOverlapClust50DPCASilvClock + as.numeric(Sys.time()-checkpoint)

#
		checkpoint<-Sys.time()
		naomiPCADip[i]<-dip.test(prcomp(naomiClust)$x[,1])$p.value
		naomiPCADipClock = naomiPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoOverlapClust50DPCADip[i]<-dip.test(prcomp(twoOverlapClust50D)$x[,1])$p.value
		twoOverlapClust50DPCADipClock = twoOverlapClust50DPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DPCADip[i]<-dip.test(prcomp(twoCloseClust50D)$x[,1])$p.value
		twoCloseClust50DPCADipClock = twoCloseClust50DPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DPCADip[i]<-dip.test(prcomp(singleClust50D)$x[,1])$p.value
		singleClust50DPCADipClock = singleClust50DPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCirclePCADip[i]<-dip.test(prcomp(fiveConcCircles)$x[,1])$p.value
		fiveConcCirclePCADipClock = fiveConcCirclePCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singlePCADip[i]<-dip.test(prcomp(singleClust)$x[,1])$p.value
		singlePCADipClock = singlePCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DPCADip[i]<-dip.test(prcomp(singleClust3D)$x[,1])$p.value
		single3DPCADipClock = single3DPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DPCADip[i]<-dip.test(prcomp(singleClust10D)$x[,1])$p.value
		single10DPCADipClock = single10DPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierPCADip[i]<-dip.test(prcomp(singleClustOutlier)$x[,1])$p.value
		singleOutlierPCADipClock = singleOutlierPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierPCADip[i]<-dip.test(prcomp(largeClustOutlier)$x[,1])$p.value
		largeOutlierPCADipClock = largeOutlierPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersPCADip[i]<-dip.test(prcomp(singleClust3Outliers)$x[,1])$p.value
		single3OutliersPCADipClock = single3OutliersPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepPCADip[i]<-dip.test(prcomp(twoSepClust)$x[,1])$p.value
		twoSepPCADipClock = twoSepPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepPCADip[i]<-dip.test(prcomp(threeSepClust)$x[,1])$p.value
		threeSepPCADipClock = threeSepPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DPCADip[i]<-dip.test(prcomp(threeSepClust3D)$x[,1])$p.value
		threeSep3DPCADipClock = threeSep3DPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeClosePCADip[i]<-dip.test(prcomp(threeCloseClust)$x[,1])$p.value
		threeClosePCADipClock = threeClosePCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyPCADip[i]<-dip.test(prcomp(threeNoisyClust)$x[,1])$p.value
		threeNoisyPCADipClock = threeNoisyPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterPCADip[i]<-dip.test(prcomp(threeClustDiffDiameter)$x[,1])$p.value
		threeDiffDiameterPCADipClock = threeDiffDiameterPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityPCADip[i]<-dip.test(prcomp(threeClustDiffDensity)$x[,1])$p.value
		threeDiffDensityPCADipClock = threeDiffDensityPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DPCADip[i]<-dip.test(prcomp(twoSepClust10D)$x[,1])$p.value
		twoSep10DPCADipClock = twoSep10DPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DPCADip[i]<-dip.test(prcomp(fourSepClust10D)$x[,1])$p.value
		fourSep10DPCADipClock = fourSep10DPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedPCADip[i]<-dip.test(prcomp(tenSeparatedClusters)$x[,1])$p.value
		tenSeparatedPCADipClock = tenSeparatedPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenClosePCADip[i]<-dip.test(prcomp(tenCloseClusters)$x[,1])$p.value
		tenClosePCADipClock = tenClosePCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDensePCADip[i]<-dip.test(prcomp(sevenCloseDenseClusters)$x[,1])$p.value		
		sevenCloseDensePCADipClock = sevenCloseDensePCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveClosePCADip[i]<-dip.test(prcomp(fiveCloseClusters)$x[,1])$p.value
		fiveClosePCADipClock = fiveClosePCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenClosePCADip[i]<-dip.test(prcomp(sevenCloseClusters)$x[,1])$p.value
		sevenClosePCADipClock = sevenClosePCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesPCADip[i]<-dip.test(prcomp(twoSepLines)$x[,1])$p.value
		parallelLinesPCADipClock =parallelLinesPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesPCADip[i]<-dip.test(prcomp(concCircles)$x[,1])$p.value
		concCirclesPCADipClock = concCirclesPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesPCADip[i]<-dip.test(prcomp(threeConcCircles)$x[,1])$p.value
		threeConcCirclesPCADipClock = threeConcCirclesPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfPCADip[i]<-dip.test(prcomp(oneTclust10df)$x[,1])$p.value
		oneTclust10dfPCADipClock = oneTclust10dfPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfPCADip[i]<-dip.test(prcomp(twoTclust10df)$x[,1])$p.value
		twoTclust10dfPCADipClock = twoTclust10dfPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfPCADip[i]<-dip.test(prcomp(oneTclust5df)$x[,1])$p.value
		oneTclust5dfPCADipClock = oneTclust5dfPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfPCADip[i]<-dip.test(prcomp(twoTclust5df)$x[,1])$p.value
		twoTclust5dfPCADipClock = twoTclust5dfPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfPCADip[i]<-dip.test(prcomp(oneTclust15df)$x[,1])$p.value
		oneTclust15dfPCADipClock = oneTclust15dfPCADipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfPCADip[i]<-dip.test(prcomp(twoTclust15df)$x[,1])$p.value
		twoTclust15dfPCADipClock = twoTclust15dfPCADipClock + as.numeric(Sys.time()-checkpoint)




	#PRINCIPAL CURVE
		pcit=100

		checkpoint<-Sys.time()
		naomiPrincurve<-principal.curve(naomiClust, maxit=pcit)
		naomiPrincurveDipClock = naomiPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		naomiPrincurveSilvClock = naomiPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)


		checkpoint<-Sys.time()
		singleClust50DPrincurve<-principal.curve(singleClust50D, maxit=pcit)
		singleClust50DPrincurveDipClock = singleClust50DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		ingleClust50DPrincurveSilvClock = singleClust50DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCirclePrincurve<-principal.curve(fiveConcCircles, maxit=pcit)
		fiveConcCirclePrincurveDipClock = fiveConcCirclePrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		fiveConcCirclePrincurveSilvClock = fiveConcCirclePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singlePrincurve<-principal.curve(singleClust, maxit=pcit)
		singlePrincurveDipClock = singlePrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		singlePrincurveSilvClock = singlePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DPrincurve<-principal.curve(singleClust3D, maxit=pcit)
		single3DPrincurveDipClock = single3DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		single3DPrincurveSilvClock = single3DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DPrincurve<-principal.curve(singleClust10D, maxit=pcit)
		single10DPrincurveDipClock = single10DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		single10DPrincurveSilvClock = single10DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierPrincurve<-principal.curve(singleClustOutlier, maxit=pcit)
		singleOutlierPrincurveDipClock = singleOutlierPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		singleOutlierPrincurveSilvClock = singleOutlierPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierPrincurve<-principal.curve(largeClustOutlier, maxit=pcit)
		largeOutlierPrincurveDipClock = largeOutlierPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		largeOutlierPrincurveSilvClock = largeOutlierPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersPrincurve<-principal.curve(singleClust3Outliers, maxit=pcit)
		single3OutliersPrincurveDipClock = single3OutliersPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		single3OutliersPrincurveSilvClock = single3OutliersPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)
	
		checkpoint<-Sys.time()
		twoSepPrincurve<-principal.curve(twoSepClust, maxit=pcit)
		twoSepPrincurveDipClock = twoSepPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		twoSepPrincurveSilvClock = twoSepPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepPrincurve<-principal.curve(threeSepClust, maxit=pcit)
		threeSepPrincurveDipClock = threeSepPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		threeSepPrincurveSilvClock = threeSepPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DPrincurve<-principal.curve(threeSepClust3D, maxit=pcit)
		threeSep3DPrincurveDipClock = threeSep3DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		threeSep3DPrincurveSilvClock = threeSep3DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeClosePrincurve<-principal.curve(threeCloseClust, maxit=pcit)
		threeClosePrincurveDipClock = threeClosePrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		threeClosePrincurveSilvClock = threeClosePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyPrincurve<-principal.curve(threeNoisyClust, maxit=pcit)
		threeNoisyPrincurveDipClock = threeNoisyPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		threeNoisyPrincurveSilvClock = threeNoisyPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterPrincurve<-principal.curve(threeClustDiffDiameter, maxit=pcit)
		threeDiffDiameterPrincurveDipClock = threeDiffDiameterPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		threeDiffDiameterPrincurveSilvClock = threeDiffDiameterPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityPrincurve<-principal.curve(threeClustDiffDensity, maxit=pcit)
		threeDiffDensityPrincurveDipClock = threeDiffDensityPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		threeDiffDensityPrincurveSilvClock = threeDiffDensityPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DPrincurve<-principal.curve(twoSepClust10D, maxit=pcit)
		twoSep10DPrincurveDipClock = twoSep10DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		twoSep10DPrincurveSilvClock = twoSep10DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DPrincurve<-principal.curve(fourSepClust10D, maxit=pcit)
		fourSep10DPrincurveDipClock = fourSep10DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		fourSep10DPrincurveSilvClock = fourSep10DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedPrincurve<-principal.curve(tenSeparatedClusters, maxit=pcit)
		tenSeparatedPrincurveDipClock = tenSeparatedPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		tenSeparatedPrincurveSilvClock = tenSeparatedPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenClosePrincurve<-principal.curve(tenCloseClusters, maxit=pcit)
		tenClosePrincurveDipClock = tenClosePrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		tenClosePrincurveSilvClock = tenClosePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDensePrincurve<-principal.curve(sevenCloseDenseClusters, maxit=pcit)
		sevenCloseDensePrincurveDipClock = sevenCloseDensePrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		sevenCloseDensePrincurveSilvClock = sevenCloseDensePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveClosePrincurve<-principal.curve(fiveCloseClusters, maxit=pcit)
		fiveClosePrincurveDipClock = fiveClosePrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		fiveClosePrincurveSilvClock = fiveClosePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenClosePrincurve<-principal.curve(sevenCloseClusters, maxit=pcit)
		sevenClosePrincurveDipClock = sevenClosePrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		sevenClosePrincurveSilvClock = sevenClosePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesPrincurve<-principal.curve(twoSepLines, maxit=pcit)
		parallelLinesPrincurveDipClock = parallelLinesPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		parallelLinesPrincurveSilvClock = parallelLinesPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesPrincurve<-principal.curve(concCircles, maxit=pcit)
		concCirclesPrincurveDipClock = concCirclesPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		concCirclesPrincurveSilvClock = concCirclesPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesPrincurve<-principal.curve(threeConcCircles, maxit=pcit)
		threeConcCirclesPrincurveDipClock = threeConcCirclesPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		threeConcCirclesPrincurveSilvClock = threeConcCirclesPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfPrincurve<-principal.curve(oneTclust10df, maxit=pcit)
		oneTclust10dfPrincurveDipClock = oneTclust10dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		oneTclust10dfPrincurveSilvClock = oneTclust10dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfPrincurve<-principal.curve(twoTclust10df, maxit=pcit)
		twoTclust10dfPrincurveDipClock = twoTclust10dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		twoTclust10dfPrincurveSilvClock = twoTclust10dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfPrincurve<-principal.curve(oneTclust5df, maxit=pcit)
		oneTclust5dfPrincurveDipClock = oneTclust5dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		oneTclust5dfPrincurveSilvClock = oneTclust5dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfPrincurve<-principal.curve(twoTclust5df, maxit=pcit)
		twoTclust5dfPrincurveDipClock = twoTclust5dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		twoTclust5dfPrincurveSilvClock = twoTclust5dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfPrincurve<-principal.curve(oneTclust15df, maxit=pcit)
		oneTclust15dfPrincurveDipClock = oneTclust15dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		oneTclust15dfPrincurveSilvClock = oneTclust15dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfPrincurve<-principal.curve(twoTclust15df, maxit=pcit)
		twoTclust15dfPrincurveDipClock = twoTclust15dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		twoTclust15dfPrincurveSilvClock = twoTclust15dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoOverlapClust50DPrincurve<-principal.curve(twoOverlapClust50D, maxit=pcit)
		twoOverlapClust50DPrincurveDipClock = twoOverlapClust50DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		twoOverlapClust50DPrincurveSilvClock = twoOverlapClust50DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DPrincurve<-principal.curve(twoCloseClust50D, maxit=pcit)
		twoCloseClust50DPrincurveDipClock = twoCloseClust50DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		twoCloseClust50DPrincurveSilvClock = twoCloseClust50DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)



	#Dip and Silvcheck
		checkpoint<-Sys.time()
		naomiPrincurveDip[i]<-dip.test(naomiPrincurve$s[,1])$p.value
		naomiPrincurveDipClock = naomiPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DPrincurveDip[i]<-dip.test(singleClust50DPrincurve$s[,1])$p.value
		singleClust50DPrincurveDipClock = singleClust50DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		
		checkpoint<-Sys.time()
		fiveConcCirclePrincurveDip[i]<-dip.test(fiveConcCirclePrincurve$s[,1])$p.value
		fiveConcCirclePrincurveDipClock = fiveConcCirclePrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singlePrincurveDip[i]<-dip.test(singlePrincurve$s[,1])$p.value
		singlePrincurveDipClock = singlePrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DPrincurveDip[i]<-dip.test(single3DPrincurve$s[,1])$p.value
		single3DPrincurveDipClock = single3DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DPrincurveDip[i]<-dip.test(single10DPrincurve$s[,1])$p.value
		single10DPrincurveDipClock = single10DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierPrincurveDip[i]<-dip.test(singleOutlierPrincurve$s[,1])$p.value
		singleOutlierPrincurveDipClock = singleOutlierPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierPrincurveDip[i]<-dip.test(largeOutlierPrincurve$s[,1])$p.value
		largeOutlierPrincurveDipClock = largeOutlierPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersPrincurveDip[i]<-dip.test(single3OutliersPrincurve$s[,1])$p.value
		single3OutliersPrincurveDipClock = single3OutliersPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepPrincurveDip[i]<-dip.test(twoSepPrincurve$s[,1])$p.value
		twoSepPrincurveDipClock = twoSepPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepPrincurveDip[i]<-dip.test(threeSepPrincurve$s[,1])$p.value
		threeSepPrincurveDipClock = threeSepPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DPrincurveDip[i]<-dip.test(threeSep3DPrincurve$s[,1])$p.value
		threeSep3DPrincurveDipClock = threeSep3DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeClosePrincurveDip[i]<-dip.test(threeClosePrincurve$s[,1])$p.value
		threeClosePrincurveDipClock = threeClosePrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyPrincurveDip[i]<-dip.test(threeNoisyPrincurve$s[,1])$p.value
		threeNoisyPrincurveDipClock = threeNoisyPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterPrincurveDip[i]<-dip.test(threeDiffDiameterPrincurve$s[,1])$p.value
		threeDiffDiameterPrincurveDipClock = threeDiffDiameterPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityPrincurveDip[i]<-dip.test(threeDiffDensityPrincurve$s[,1])$p.value
		threeDiffDensityPrincurveDipClock = threeDiffDensityPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DPrincurveDip[i]<-dip.test(twoSep10DPrincurve$s[,1])$p.value
		twoSep10DPrincurveDipClock = twoSep10DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DPrincurveDip[i]<-dip.test(fourSep10DPrincurve$s[,1])$p.value
		fourSep10DPrincurveDipClock = fourSep10DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedPrincurveDip[i]<-dip.test(tenSeparatedPrincurve$s[,1])$p.value
		tenSeparatedPrincurveDipClock = tenSeparatedPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenClosePrincurveDip[i]<-dip.test(tenClosePrincurve$s[,1])$p.value
		tenClosePrincurveDipClock = tenClosePrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDensePrincurveDip[i]<-dip.test(sevenCloseDensePrincurve$s[,1])$p.value
		sevenCloseDensePrincurveDipClock = sevenCloseDensePrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveClosePrincurveDip[i]<-dip.test(fiveClosePrincurve$s[,1])$p.value
		fiveClosePrincurveDipClock = fiveClosePrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenClosePrincurveDip[i]<-dip.test(sevenClosePrincurve$s[,1])$p.value
		sevenClosePrincurveDipClock = sevenClosePrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesPrincurveDip[i]<-dip.test(parallelLinesPrincurve$s[,1])$p.value
		parallelLinesPrincurveDipClock = parallelLinesPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesPrincurveDip[i]<-dip.test(concCirclesPrincurve$s[,1])$p.value
		concCirclesPrincurveDipClock = concCirclesPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesPrincurveDip[i]<-dip.test(threeConcCirclesPrincurve$s[,1])$p.value
		threeConcCirclesPrincurveDipClock = threeConcCirclesPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfPrincurveDip[i]<-dip.test(oneTclust10dfPrincurve$s[,1])$p.value
		oneTclust10dfPrincurveDipClock = oneTclust10dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfPrincurveDip[i]<-dip.test(twoTclust10dfPrincurve$s[,1])$p.value
		twoTclust10dfPrincurveDipClock = twoTclust10dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfPrincurveDip[i]<-dip.test(oneTclust5dfPrincurve$s[,1])$p.value
		oneTclust5dfPrincurveDipClock = oneTclust5dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfPrincurveDip[i]<-dip.test(twoTclust5dfPrincurve$s[,1])$p.value
		twoTclust5dfPrincurveDipClock = twoTclust5dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfPrincurveDip[i]<-dip.test(oneTclust15dfPrincurve$s[,1])$p.value
		oneTclust15dfPrincurveDipClock = oneTclust15dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfPrincurveDip[i]<-dip.test(twoTclust15dfPrincurve$s[,1])$p.value
		twoTclust15dfPrincurveDipClock = twoTclust15dfPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoOverlapClust50DPrincurveDip[i]<-dip.test(twoOverlapClust50DPrincurve$s[,1])$p.value
		twoOverlapClust50DPrincurveDipClock = twoOverlapClust50DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DPrincurveDip[i]<-dip.test(twoCloseClust50DPrincurve$s[,1])$p.value
		twoCloseClust50DPrincurveDipClock = twoCloseClust50DPrincurveDipClock + as.numeric(Sys.time()-checkpoint)

#
		checkpoint<-Sys.time()
		naomiPrincurveSilv[i]<-silverman.test(naomiPrincurve$s[,1],1,adjust=TRUE)@p_value
		naomiPrincurveSilvClock = naomiPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleClust50DPrincurveSilv[i]<-silverman.test(singleClust50DPrincurve$s[,1],1,adjust=TRUE)@p_value
		singleClust50DPrincurveSilvClock = singleClust50DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveConcCirclePrincurveSilv[i]<-silverman.test(fiveConcCirclePrincurve$s[,1],1,adjust=TRUE)@p_value
		fiveConcCirclePrincurveSilvClock = fiveConcCirclePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singlePrincurveSilv[i]<-silverman.test(singlePrincurve$s[,1],1,adjust=TRUE)@p_value
		singlePrincurveSilvClock = singlePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3DPrincurveSilv[i]<-silverman.test(single3DPrincurve$s[,1],1,adjust=TRUE)@p_value
		single3DPrincurveSilvClock = single3DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single10DPrincurveSilv[i]<-silverman.test(single10DPrincurve$s[,1],1,adjust=TRUE)@p_value
		single10DPrincurveSilvClock = single10DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		singleOutlierPrincurveSilv[i]<-silverman.test(singleOutlierPrincurve$s[,1],1,adjust=TRUE)@p_value
		singleOutlierPrincurveSilvClock = singleOutlierPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		largeOutlierPrincurveSilv[i]<-silverman.test(largeOutlierPrincurve$s[,1],1,adjust=TRUE)@p_value
		largeOutlierPrincurveSilvClock = largeOutlierPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		single3OutliersPrincurveSilv[i]<-silverman.test(single3OutliersPrincurve$s[,1],1,adjust=TRUE)@p_value
		single3OutliersPrincurveSilvClock = single3OutliersPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSepPrincurveSilv[i]<-silverman.test(twoSepPrincurve$s[,1],1,adjust=TRUE)@p_value
		twoSepPrincurveSilvClock = twoSepPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSepPrincurveSilv[i]<-silverman.test(threeSepPrincurve$s[,1],1,adjust=TRUE)@p_value
		threeSepPrincurveSilvClock = threeSepPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeSep3DPrincurveSilv[i]<-silverman.test(threeSep3DPrincurve$s[,1],1,adjust=TRUE)@p_value
		threeSep3DPrincurveSilvClock = threeSep3DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeClosePrincurveSilv[i]<-silverman.test(threeClosePrincurve$s[,1],1,adjust=TRUE)@p_value
		threeClosePrincurveSilvClock = threeClosePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeNoisyPrincurveSilv[i]<-silverman.test(threeNoisyPrincurve$s[,1],1,adjust=TRUE)@p_value
		threeNoisyPrincurveSilvClock = threeNoisyPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDiameterPrincurveSilv[i]<-silverman.test(threeDiffDiameterPrincurve$s[,1],1,adjust=TRUE)@p_value
		threeDiffDiameterPrincurveSilvClock = threeDiffDiameterPrincurveSilvClock  + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeDiffDensityPrincurveSilv[i]<-silverman.test(threeDiffDensityPrincurve$s[,1],1,adjust=TRUE)@p_value
		threeDiffDensityPrincurveSilvClock = threeDiffDensityPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoSep10DPrincurveSilv[i]<-silverman.test(twoSep10DPrincurve$s[,1],1,adjust=TRUE)@p_value
		twoSep10DPrincurveSilvClock = twoSep10DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fourSep10DPrincurveSilv[i]<-silverman.test(fourSep10DPrincurve$s[,1],1,adjust=TRUE)@p_value
		fourSep10DPrincurveSilvClock = fourSep10DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenSeparatedPrincurveSilv[i]<-silverman.test(tenSeparatedPrincurve$s[,1],1,adjust=TRUE)@p_value
		tenSeparatedPrincurveSilvClock = tenSeparatedPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		tenClosePrincurveSilv[i]<-silverman.test(tenClosePrincurve$s[,1],1,adjust=TRUE)@p_value
		tenClosePrincurveSilvClock = tenClosePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenCloseDensePrincurveSilv[i]<-silverman.test(sevenCloseDensePrincurve$s[,1],1,adjust=TRUE)@p_value
		sevenCloseDensePrincurveSilvClock = sevenCloseDensePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		fiveClosePrincurveSilv[i]<-silverman.test(fiveClosePrincurve$s[,1],1,adjust=TRUE)@p_value
		fiveClosePrincurveSilvClock = fiveClosePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		sevenClosePrincurveSilv[i]<-silverman.test(sevenClosePrincurve$s[,1],1,adjust=TRUE)@p_value
		sevenClosePrincurveSilvClock = sevenClosePrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		parallelLinesPrincurveSilv[i]<-silverman.test(parallelLinesPrincurve$s[,1],1,adjust=TRUE)@p_value
		parallelLinesPrincurveSilvClock = parallelLinesPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		concCirclesPrincurveSilv[i]<-silverman.test(concCirclesPrincurve$s[,1],1,adjust=TRUE)@p_value
		concCirclesPrincurveSilvClock = concCirclesPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		threeConcCirclesPrincurveSilv[i]<-silverman.test(threeConcCirclesPrincurve$s[,1],1,adjust=TRUE)@p_value
		threeConcCirclesPrincurveSilvClock = threeConcCirclesPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust10dfPrincurveSilv[i]<-silverman.test(oneTclust10dfPrincurve$s[,1],1,adjust=TRUE)@p_value
		oneTclust10dfPrincurveSilvClock = oneTclust10dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust10dfPrincurveSilv[i]<-silverman.test(twoTclust10dfPrincurve$s[,1],1,adjust=TRUE)@p_value
		twoTclust10dfPrincurveSilvClock = twoTclust10dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust5dfPrincurveSilv[i]<-silverman.test(oneTclust5dfPrincurve$s[,1],1,adjust=TRUE)@p_value
		oneTclust5dfPrincurveSilvClock = oneTclust5dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust5dfPrincurveSilv[i]<-silverman.test(twoTclust5dfPrincurve$s[,1],1,adjust=TRUE)@p_value
		twoTclust5dfPrincurveSilvClock = twoTclust5dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		oneTclust15dfPrincurveSilv[i]<-silverman.test(oneTclust15dfPrincurve$s[,1],1,adjust=TRUE)@p_value
		oneTclust15dfPrincurveSilvClock = oneTclust15dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoTclust15dfPrincurveSilv[i]<-silverman.test(twoTclust15dfPrincurve$s[,1],1,adjust=TRUE)@p_value
		twoTclust15dfPrincurveSilvClock = twoTclust15dfPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoOverlapClust50DPrincurveSilv[i]<-silverman.test(twoOverlapClust50DPrincurve$s[,1],1,adjust=TRUE)@p_value
		twoOverlapClust50DPrincurveSilvClock = twoOverlapClust50DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		twoCloseClust50DPrincurveSilv[i]<-silverman.test(twoCloseClust50DPrincurve$s[,1],1,adjust=TRUE)@p_value
		twoCloseClust50DPrincurveSilvClock = twoCloseClust50DPrincurveSilvClock + as.numeric(Sys.time()-checkpoint)


	
	#find convergence
		naomiPrincurveConv[i]<-naomiPrincurve$converged

		singleClust50DPrincurveConv[i]<-singleClust50DPrincurve$converged
		fiveConcCirclePrincurveConv[i]<-fiveConcCirclePrincurve$converged
		singlePrincurveConv[i]<-singlePrincurve$converged
		single3DPrincurveConv[i]<-single3DPrincurve$converged
		single10DPrincurveConv[i]<-single10DPrincurve$converged
		singleOutlierPrincurveConv[i]<-singleOutlierPrincurve$converged
		largeOutlierPrincurveConv[i]<-largeOutlierPrincurve$converged
		single3OutliersPrincurveConv[i]<-single3OutliersPrincurve$converged
		twoSepPrincurveConv[i]<-twoSepPrincurve$converged
		threeSepPrincurveConv[i]<-threeSepPrincurve$converged
		threeSep3DPrincurveConv[i]<-threeSep3DPrincurve$converged
		threeClosePrincurveConv[i]<-threeClosePrincurve$converged
		threeNoisyPrincurveConv[i]<-threeNoisyPrincurve$converged
		threeDiffDiameterPrincurveConv[i]<-threeDiffDiameterPrincurve$converged
		threeDiffDensityPrincurveConv[i]<-threeDiffDensityPrincurve$converged
		twoSep10DPrincurveConv[i]<-twoSep10DPrincurve$converged
		fourSep10DPrincurveConv[i]<-fourSep10DPrincurve$converged
		tenSeparatedPrincurveConv[i]<-tenSeparatedPrincurve$converged
		tenClosePrincurveConv[i]<-tenClosePrincurve$converged
		sevenCloseDensePrincurveConv[i]<-sevenCloseDensePrincurve$converged
		fiveClosePrincurveConv[i]<-fiveClosePrincurve$converged
		sevenClosePrincurveConv[i]<-sevenClosePrincurve$converged
		parallelLinesPrincurveConv[i]<-parallelLinesPrincurve$converged
		concCirclesPrincurveConv[i]<-concCirclesPrincurve$converged
		threeConcCirclesPrincurveConv[i]<-threeConcCirclesPrincurve$converged
		oneTclust10dfPrincurveConv[i]<-oneTclust10dfPrincurve$converged
		twoTclust10dfPrincurveConv[i]<-twoTclust10dfPrincurve$converged
		oneTclust5dfPrincurveConv[i]<-oneTclust5dfPrincurve$converged
		twoTclust5dfPrincurveConv[i]<-twoTclust5dfPrincurve$converged
		oneTclust15dfPrincurveConv[i]<-oneTclust15dfPrincurve$converged
		twoTclust15dfPrincurveConv[i]<-twoTclust15dfPrincurve$converged
		twoOverlapClust50DPrincurveConv[i]<-twoOverlapClust50DPrincurve$converged
		twoCloseClust50DPrincurveConv[i]<-twoCloseClust50DPrincurve$converged		


	#feedback
		loopClock<-as.numeric(proc.time()-loopTimer)[3]		#stop loopstopwatch, and extract elapsed time
		clock=clock+loopClock						#add to total time passed.
		clockRemain=((3*(clock/i)+(loopClock))/4)*(it-i)		#predict future based on average of current simulation speed vs history

		cat(" Finished Generation: ", i, "/", it)
		printTime(prefix=" in ", loopClock)
		printTime(prefix="Total Time Elapsed: ", clock)  
		printTime(prefix="Estimated Time Remaining: ",clockRemain,suffix="\n")	
		flush.console()
		Sys.sleep(0.001)
	}

	timer<-matrix(,34,9)
	result<-matrix(,34,15)


	print("iterations:")
	print(it)
	print(singleDipClock)
	print(singleSilvClock)
	print(round(singleHopClock,4))
	print(singleClassicDipClock)
	print(singleClassicSilvClock)
	print(singlePCADipClock)
	print(singlePCASilvClock)
	print(round(singlePrincurveDipClock,4))
	print(round(singlePrincurveSilvClock,4))

	timer[1,1]<-singleDipClock/it
	timer[1,2]<-singleSilvClock/it
	timer[1,3]<-round(singleHopClock/it,4)
	timer[1,4]<-singleClassicDipClock/it
	timer[1,5]<-singleClassicSilvClock/it
	timer[1,6]<-singlePCADipClock/it
	timer[1,7]<-singlePCASilvClock/it
	timer[1,8]<-round(singlePrincurveDipClock/it,4)
	timer[1,9]<-round(singlePrincurveSilvClock/it,4)
	timer[2,1]<-single3DDipClock/it
	timer[2,2]<-single3DSilvClock/it
	timer[2,3]<-round(single3DHopClock/it,4)
	timer[2,4]<-single3DClassicDipClock/it
	timer[2,5]<-single3DClassicSilvClock/it
	timer[2,6]<-single3DPCADipClock/it
	timer[2,7]<-single3DPCASilvClock/it
	timer[2,8]<-round(single3DPrincurveDipClock/it,4)
	timer[2,9]<-round(single3DPrincurveSilvClock/it,4)
	timer[3,1]<-single10DDipClock/it
	timer[3,2]<-single10DSilvClock/it
	timer[3,3]<-round(single10DHopClock/it,4)
	timer[3,4]<-single10DClassicDipClock/it	
	timer[3,5]<-single10DClassicSilvClock/it
	timer[3,6]<-single10DPCADipClock/it
	timer[3,7]<-single10DPCASilvClock/it
	timer[3,8]<-round(single10DPrincurveDipClock/it,4)
	timer[3,9]<-round(single10DPrincurveSilvClock/it,4)

	timer[4,1]<-singleClust50DDipClock/it
	timer[4,2]<-singleClust50DSilvClock/it
	timer[4,3]<-round(singleClust50DHopClock/it,4)
	timer[4,4]<-singleClust50DClassicDipClock/it
	timer[4,5]<-singleClust50DClassicSilvClock/it
	timer[4,6]<-singleClust50DPCADipClock/it
	timer[4,7]<-singleClust50DPCASilvClock/it
	timer[4,8]<-round(singleClust50DPrincurveDipClock/it,4)
	timer[4,9]<-round(singleClust50DPrincurveSilvClock/it,4)
	timer[5,1]<-singleOutlierDipClock/it
	timer[5,2]<-singleOutlierSilvClock/it
	timer[5,3]<-round(singleOutlierHopClock/it,4)
	timer[5,4]<-singleOutlierClassicDipClock/it
	timer[5,5]<-singleOutlierClassicSilvClock/it
	timer[5,6]<-singleOutlierPCADipClock/it
	timer[5,7]<-singleOutlierPCASilvClock/it
	timer[5,8]<-round(singleOutlierPrincurveDipClock/it,4)
	timer[5,9]<-round(singleOutlierPrincurveSilvClock/it,4)
	timer[6,1]<-largeOutlierDipClock/it
	timer[6,2]<-largeOutlierSilvClock/it
	timer[6,3]<-round(largeOutlierHopClock/it,4)
	timer[6,4]<-largeOutlierClassicDipClock/it
	timer[6,5]<-largeOutlierClassicSilvClock/it
	timer[6,6]<-largeOutlierPCADipClock/it
	timer[6,7]<-largeOutlierPCASilvClock/it
	timer[6,8]<-round(largeOutlierPrincurveDipClock/it,4)
	timer[6,9]<-round(largeOutlierPrincurveSilvClock/it,4)
	timer[7,1]<-single3OutliersDipClock/it
	timer[7,2]<-single3OutliersSilvClock/it
	timer[7,3]<-round(single3OutliersHopClock/it,4)
	timer[7,4]<-single3OutliersClassicDipClock/it
	timer[7,5]<-single3OutliersClassicSilvClock/it
	timer[7,6]<-single3OutliersPCADipClock/it
	timer[7,7]<-single3OutliersPCASilvClock/it
	timer[7,8]<-round(single3OutliersPrincurveDipClock/it,4)
	timer[7,9]<-round(single3OutliersPrincurveSilvClock/it,4)
	timer[8,1]<-oneTclust5dfDipClock/it
	timer[8,2]<-oneTclust5dfSilvClock/it
	timer[8,3]<-round(oneTclust5dfHopClock/it,4)
	timer[8,4]<-oneTclust5dfClassicDipClock/it
	timer[8,5]<-oneTclust5dfClassicSilvClock/it
	timer[8,6]<-oneTclust5dfPCADipClock/it
	timer[8,7]<-oneTclust5dfPCASilvClock/it
	timer[8,8]<-round(oneTclust5dfPrincurveDipClock/it,4)
	timer[8,9]<-round(oneTclust5dfPrincurveSilvClock/it,4)
	timer[9,1]<-oneTclust10dfDipClock/it
	timer[9,2]<-oneTclust10dfSilvClock/it
	timer[9,3]<-round(oneTclust10dfHopClock/it,4)
	timer[9,4]<-oneTclust10dfClassicDipClock/it
	timer[9,5]<-oneTclust10dfClassicSilvClock/it
	timer[9,6]<-oneTclust10dfPCADipClock/it
	timer[9,7]<-oneTclust10dfPCASilvClock/it
	timer[9,8]<-round(oneTclust10dfPrincurveDipClock/it,4)
	timer[9,9]<-round(oneTclust10dfPrincurveSilvClock/it,4)
	timer[10,1]<-oneTclust15dfDipClock/it
	timer[10,2]<-oneTclust15dfSilvClock/it
	timer[10,3]<-round(oneTclust15dfHopClock/it,4)
	timer[10,4]<-oneTclust15dfClassicDipClock/it
	timer[10,5]<-oneTclust15dfClassicSilvClock/it
	timer[10,6]<-oneTclust15dfPCADipClock/it
	timer[10,7]<-oneTclust15dfPCASilvClock/it
	timer[10,8]<-round(oneTclust15dfPrincurveDipClock/it,4)
	timer[10,9]<-round(oneTclust15dfPrincurveSilvClock/it,4)

#

	timer[11,1]<-twoSepDipClock/it
	timer[11,2]<-twoSepSilvClock/it
	timer[11,3]<-round(twoSepHopClock/it,4)
	timer[11,4]<-twoSepClassicDipClock/it
	timer[11,5]<-twoSepClassicSilvClock/it
	timer[11,6]<-twoSepPCADipClock/it
	timer[11,7]<-twoSepPCASilvClock/it
	timer[11,8]<-round(twoSepPrincurveDipClock/it,4)
	timer[11,9]<-round(twoSepPrincurveSilvClock/it,4)
	timer[12,1]<-threeSepDipClock/it
	timer[12,2]<-threeSepSilvClock/it
	timer[12,3]<-round(threeSepHopClock/it,4)
	timer[12,4]<-threeSepClassicDipClock/it
	timer[12,5]<-threeSepClassicSilvClock/it
	timer[12,6]<-threeSepPCADipClock/it
	timer[12,7]<-threeSepPCASilvClock/it
	timer[12,8]<-round(threeSepPrincurveDipClock/it,4)
	timer[12,9]<-round(threeSepPrincurveSilvClock/it,4)
	timer[13,1]<-threeSep3DDipClock/it
	timer[13,2]<-threeSep3DSilvClock/it
	timer[13,3]<-round(threeSep3DHopClock/it,4)
	timer[13,4]<-threeSep3DClassicDipClock/it
	timer[13,5]<-threeSep3DClassicSilvClock/it
	timer[13,6]<-threeSep3DPCADipClock/it
	timer[13,7]<-threeSep3DPCASilvClock/it
	timer[13,8]<-round(threeSep3DPrincurveDipClock/it,4)
	timer[13,9]<-round(threeSep3DPrincurveSilvClock/it,4)
	timer[14,1]<-threeCloseDipClock/it
	timer[14,2]<-threeCloseSilvClock/it
	timer[14,3]<-round(threeCloseHopClock/it,4)
	timer[14,4]<-threeCloseClassicDipClock/it
	timer[14,5]<-threeCloseClassicSilvClock/it
	timer[14,6]<-threeClosePCADipClock/it
	timer[14,7]<-threeClosePCASilvClock/it
	timer[14,8]<-round(threeClosePrincurveDipClock/it,4)
	timer[14,9]<-round(threeClosePrincurveSilvClock/it,4)
	timer[15,1]<-threeNoisyDipClock/it
	timer[15,2]<-threeNoisySilvClock/it
	timer[15,3]<-round(threeNoisyHopClock/it,4)
	timer[15,4]<-threeNoisyClassicDipClock/it
	timer[15,5]<-threeNoisyClassicSilvClock/it
	timer[15,6]<-threeNoisyPCADipClock/it
	timer[15,7]<-threeNoisyPCASilvClock/it
	timer[15,8]<-round(threeNoisyPrincurveDipClock/it,4)
	timer[15,9]<-round(threeNoisyPrincurveSilvClock/it,4)
	timer[16,1]<-threeDiffDiameterDipClock/it
	timer[16,2]<-threeDiffDiameterSilvClock/it
	timer[16,3]<-round(threeDiffDiameterHopClock/it,4)
	timer[16,4]<-threeDiffDiameterClassicDipClock/it
	timer[16,5]<-threeDiffDiameterClassicSilvClock/it
	timer[16,6]<-threeDiffDiameterPCADipClock/it
	timer[16,7]<-threeDiffDiameterPCASilvClock/it
	timer[16,8]<-round(threeDiffDiameterPrincurveDipClock/it,4)
	timer[16,9]<-round(threeDiffDiameterPrincurveSilvClock/it,4)
	timer[17,1]<-threeDiffDensityDipClock/it
	timer[17,2]<-threeDiffDensitySilvClock/it
	timer[17,3]<-round(threeDiffDensityHopClock/it,4)
	timer[17,4]<-threeDiffDensityClassicDipClock/it
	timer[17,5]<-threeDiffDensityClassicSilvClock/it
	timer[17,6]<-threeDiffDensityPCADipClock/it
	timer[17,7]<-threeDiffDensityPCASilvClock/it
	timer[17,8]<-round(threeDiffDensityPrincurveDipClock/it,4)
	timer[17,9]<-round(threeDiffDensityPrincurveSilvClock/it,4)
	timer[18,1]<-twoSep10DDipClock/it
	timer[18,2]<-twoSep10DSilvClock/it
	timer[18,3]<-round(twoSep10DHopClock/it,4)
	timer[18,4]<-twoSep10DClassicDipClock/it
	timer[18,5]<-twoSep10DClassicSilvClock/it
	timer[18,6]<-twoSep10DPCADipClock/it
	timer[18,7]<-twoSep10DPCASilvClock/it
	timer[18,8]<-round(twoSep10DPrincurveDipClock/it,4)
	timer[18,9]<-round(twoSep10DPrincurveSilvClock/it,4)
	timer[19,1]<-fourSep10DDipClock/it
	timer[19,2]<-fourSep10DSilvClock/it
	timer[19,3]<-round(fourSep10DHopClock/it,4)
	timer[19,4]<-fourSep10DClassicDipClock/it
	timer[19,5]<-fourSep10DClassicSilvClock/it
	timer[19,6]<-fourSep10DPCADipClock/it
	timer[19,7]<-fourSep10DPCASilvClock/it
	timer[19,8]<-round(fourSep10DPrincurveDipClock/it,4)
	timer[19,9]<-round(fourSep10DPrincurveSilvClock/it,4)
	timer[20,1]<-tenSeparatedDipClock/it
	timer[20,2]<-tenSeparatedSilvClock/it
	timer[20,3]<-round(tenSeparatedHopClock/it,4)
	timer[20,4]<-tenSeparatedClassicDipClock/it
	timer[20,5]<-tenSeparatedClassicSilvClock/it
	timer[20,6]<-tenSeparatedPCADipClock/it
	timer[20,7]<-tenSeparatedPCASilvClock/it
	timer[20,8]<-round(tenSeparatedPrincurveDipClock/it,4)
	timer[20,9]<-round(tenSeparatedPrincurveSilvClock/it,4)
	timer[21,1]<-tenCloseDipClock/it
	timer[21,2]<-tenCloseSilvClock/it
	timer[21,3]<-round(tenCloseHopClock/it,4)
	timer[21,4]<-tenCloseClassicDipClock/it
	timer[21,5]<-tenCloseClassicSilvClock/it
	timer[21,6]<-tenClosePCADipClock/it
	timer[21,7]<-tenClosePCASilvClock/it
	timer[21,8]<-round(tenClosePrincurveDipClock/it,4)
	timer[21,9]<-round(tenClosePrincurveSilvClock/it,4)
	timer[22,1]<-sevenCloseDenseDipClock/it
	timer[22,2]<-sevenCloseDenseSilvClock/it
	timer[22,3]<-round(sevenCloseDenseHopClock/it,4)
	timer[22,4]<-sevenCloseDenseClassicDipClock/it
	timer[22,5]<-sevenCloseDenseClassicSilvClock/it
	timer[22,6]<-sevenCloseDensePCADipClock/it
	timer[22,7]<-sevenCloseDensePCASilvClock/it
	timer[22,8]<-round(sevenCloseDensePrincurveDipClock/it,4)
	timer[22,9]<-round(sevenCloseDensePrincurveSilvClock/it,4)
	timer[23,1]<-sevenCloseDipClock/it
	timer[23,2]<-sevenCloseSilvClock/it
	timer[23,3]<-round(sevenCloseHopClock/it,4)
	timer[23,4]<-sevenCloseClassicDipClock/it
	timer[23,5]<-sevenCloseClassicSilvClock/it
	timer[23,6]<-sevenClosePCADipClock/it
	timer[23,7]<-sevenClosePCASilvClock/it
	timer[23,8]<-round(sevenClosePrincurveDipClock/it,4)
	timer[23,9]<-round(sevenClosePrincurveSilvClock/it,4)
	timer[24,1]<-fiveCloseDipClock/it
	timer[24,2]<-fiveCloseSilvClock/it
	timer[24,3]<-round(fiveCloseHopClock/it,4)
	timer[24,4]<-fiveCloseClassicDipClock/it
	timer[24,5]<-fiveCloseClassicSilvClock/it
	timer[24,6]<-fiveClosePCADipClock/it
	timer[24,7]<-fiveClosePCASilvClock/it
	timer[24,8]<-round(fiveClosePrincurveDipClock/it,4)
	timer[24,9]<-round(fiveClosePrincurveSilvClock/it,4)
	timer[25,1]<-parallelLinesDipClock/it
	timer[25,2]<-parallelLinesSilvClock/it
	timer[25,3]<-round(parallelLinesHopClock/it,4)
	timer[25,4]<-parallelLinesClassicDipClock/it
	timer[25,5]<-parallelLinesClassicSilvClock/it	
	timer[25,6]<-parallelLinesPCADipClock/it
	timer[25,7]<-parallelLinesPCASilvClock/it
	timer[25,8]<-round(parallelLinesPrincurveDipClock/it,4)
	timer[25,9]<-round(parallelLinesPrincurveSilvClock/it,4)
	timer[26,1]<-concCirclesDipClock/it
	timer[26,2]<-concCirclesSilvClock/it
	timer[26,3]<-round(concCirclesHopClock/it,4)
	timer[26,4]<-concCirclesClassicDipClock/it
	timer[26,5]<-concCirclesClassicSilvClock/it
	timer[26,6]<-concCirclesPCADipClock/it
	timer[26,7]<-concCirclesPCASilvClock/it
	timer[26,8]<-round(concCirclesPrincurveDipClock/it,4)
	timer[26,9]<-round(concCirclesPrincurveSilvClock/it,4)
	timer[27,1]<-threeConcCirclesDipClock/it
	timer[27,2]<-threeConcCirclesSilvClock/it
	timer[27,3]<-round(threeConcCirclesHopClock/it,4)
	timer[27,4]<-threeConcCirclesClassicDipClock/it
	timer[27,5]<-threeConcCirclesClassicSilvClock/it
	timer[27,6]<-threeConcCirclesPCADipClock/it
	timer[27,7]<-threeConcCirclesPCASilvClock/it
	timer[27,8]<-round(threeConcCirclesPrincurveDipClock/it,4)
	timer[27,9]<-round(threeConcCirclesPrincurveSilvClock/it,4)
	timer[28,1]<-fiveConcCircleDipClock/it
	timer[28,2]<-fiveConcCircleSilvClock/it
	timer[28,3]<-round(fiveConcCircleHopClock/it,4)
	timer[28,4]<-fiveConcCircleClassicDipClock/it
	timer[28,5]<-fiveConcCircleClassicSilvClock/it
	timer[28,6]<-fiveConcCirclePCADipClock/it
	timer[28,7]<-fiveConcCirclePCASilvClock/it
	timer[28,8]<-round(fiveConcCirclePrincurveDipClock/it,4)
	timer[28,9]<-round(fiveConcCirclePrincurveSilvClock/it,4)
	timer[29,1]<-twoTclust10dfDipClock/it
	timer[29,2]<-twoTclust10dfSilvClock/it
	timer[29,3]<-round(twoTclust10dfHopClock/it,4)
	timer[29,4]<-twoTclust10dfClassicDipClock/it
	timer[29,5]<-twoTclust10dfClassicSilvClock/it
	timer[29,6]<-twoTclust10dfPCADipClock/it
	timer[29,7]<-twoTclust10dfPCASilvClock/it
	timer[29,8]<-round(twoTclust10dfPrincurveDipClock/it,4)
	timer[29,9]<-round(twoTclust10dfPrincurveSilvClock/it,4)
	timer[30,1]<-twoTclust5dfDipClock/it
	timer[30,2]<-twoTclust5dfSilvClock/it
	timer[30,3]<-round(twoTclust5dfHopClock/it,4)
	timer[30,4]<-twoTclust5dfClassicDipClock/it
	timer[30,5]<-twoTclust5dfClassicSilvClock/it
	timer[30,6]<-twoTclust5dfPCADipClock/it
	timer[30,7]<-twoTclust5dfPCASilvClock/it
	timer[30,8]<-round(twoTclust5dfPrincurveDipClock/it,4)
	timer[30,9]<-round(twoTclust5dfPrincurveSilvClock/it,4)
	timer[31,1]<-twoTclust15dfDipClock/it
	timer[31,2]<-twoTclust15dfSilvClock/it
	timer[31,3]<-round(twoTclust15dfHopClock/it,4)
	timer[31,4]<-twoTclust15dfClassicDipClock/it
	timer[31,5]<-twoTclust15dfClassicSilvClock/it
	timer[31,6]<-twoTclust15dfPCADipClock/it
	timer[31,7]<-twoTclust15dfPCASilvClock/it
	timer[31,8]<-round(twoTclust15dfPrincurveDipClock/it,4)
	timer[31,9]<-round(twoTclust15dfPrincurveSilvClock/it,4)
	timer[32,1]<-twoOverlapClust50DDipClock/it
	timer[32,2]<-twoOverlapClust50DSilvClock/it
	timer[32,3]<-round(twoOverlapClust50DHopClock/it,4)
	timer[32,4]<-twoOverlapClust50DClassicDipClock/it
	timer[32,5]<-twoOverlapClust50DClassicSilvClock/it
	timer[32,6]<-twoOverlapClust50DPCADipClock/it
	timer[32,7]<-twoOverlapClust50DPCASilvClock/it
	timer[32,8]<-round(twoOverlapClust50DPrincurveDipClock/it,4)
	timer[32,9]<-round(twoOverlapClust50DPrincurveSilvClock/it,4)
	timer[33,1]<-twoCloseClust50DDipClock/it
	timer[33,2]<-twoCloseClust50DSilvClock/it
	timer[33,3]<-round(twoCloseClust50DHopClock/it,4)
	timer[33,4]<-twoCloseClust50DClassicDipClock/it
	timer[33,5]<-twoCloseClust50DClassicSilvClock/it
	timer[33,6]<-twoCloseClust50DPCADipClock/it
	timer[33,7]<-twoCloseClust50DPCASilvClock/it
	timer[33,8]<-round(twoCloseClust50DPrincurveDipClock/it,4)
	timer[33,9]<-round(twoCloseClust50DPrincurveSilvClock/it,4)

	timer[34,1]<-naomiDipClock/it
	timer[34,2]<-naomiSilvClock/it
	timer[34,3]<-round(naomiHopClock/it,4)
	timer[34,4]<-naomiClassicDipClock/it
	timer[34,5]<-naomiClassicSilvClock/it
	timer[34,6]<-naomiPCADipClock/it
	timer[34,7]<-naomiPCASilvClock/it
	timer[34,8]<-round(naomiPrincurveDipClock/it,4)
	timer[34,9]<-round(naomiPrincurveSilvClock/it,4)


	colnames(timer)<-c("Dip","Silv.","Hop","Classic Dip","Classic Silv.","PCA Dip", "PCA Silv.","Pcurv.Dip","Pcurv.Silv")
	rownames(timer)<-c("single Cluster","single Cluster 3D","single Cluster 10D","single Cluster 50D",
					"singleClustOutlier","largeClustOutlier","singleClust3Outliers","singleTcluster5df",
					"singleTcluster10df","singleTcluster15df",
		# multi-clusters:
					"twoSepClust","threeSepClust","threeSepClust3D","threeCloseClust","threeNoisyClust",
					"threeClustDiffDiameter","threeClustDiffDensity","twoSepClust10D","fourSepClust10D",
					"tenSeparatedClusters","tenCloseClusters","sevenCloseDenseClusters","sevenCloseClusters",
					"fiveCloseClusters","ParallelLines","twoConcentricCircles","threeConcentricCircles",
					"fiveConcentricCircles","twoTclusters5df","twoTclusters10df","twoTclusters15df",
					"twoOverlappingClust50D","twoCloseClust50D",
					"Naomis Cluster")



	result[1,1]<-sum(singleDip<0.05)/it
	result[1,2]<-sum(singleSilv<0.05)/it
	result[1,3]<-round(mean(singleHop),4)
	result[1,4]<-round(sum(singleHop<0.25)/it,4)
	result[1,5]<-round(sum(singleHop<qbeta(0.025, nrow(singleClust)/10, nrow(singleClust)/10))/it,4)
	result[1,6]<-round(sum(singleHop<qbeta(0.05, nrow(singleClust)/10, nrow(singleClust)/10))/it,4)
	result[1,7]<-round(sum(singleHop>qbeta(0.95, nrow(singleClust)/10, nrow(singleClust)/10))/it,4)
	result[1,8]<-round(sum(singleHop>qbeta(0.975, nrow(singleClust)/10, nrow(singleClust)/10))/it,4)
	result[1,9]<-sum(singleClassicDip<0.05)/it
	result[1,10]<-sum(singleClassicSilv<0.05)/it
	result[1,11]<-sum(singlePCADip<0.05)/it
	result[1,12]<-sum(singlePCASilv<0.05)/it
	result[1,13]<-mean(singlePrincurveConv)
	result[1,14]<-round(sum(singlePrincurveDip<0.05)/it,4)
	result[1,15]<-round(sum(singlePrincurveSilv<0.05)/it,4)
	result[2,1]<-sum(single3DDip<0.05)/it
	result[2,2]<-sum(single3DSilv<0.05)/it
	result[2,3]<-round(mean(single3DHop),4)
	result[2,4]<-round(sum(single3DHop<0.25)/it,4)
	result[2,5]<-round(sum(single3DHop<qbeta(0.025, nrow(singleClust3D)/10, nrow(singleClust3D)/10))/it,4)
	result[2,6]<-round(sum(single3DHop<qbeta(0.05, nrow(singleClust3D)/10, nrow(singleClust3D)/10))/it,4)
	result[2,7]<-round(sum(single3DHop>qbeta(0.95, nrow(singleClust3D)/10, nrow(singleClust3D)/10))/it,4)
	result[2,8]<-round(sum(single3DHop>qbeta(0.975, nrow(singleClust3D)/10, nrow(singleClust3D)/10))/it,4)
	result[2,9]<-sum(single3DClassicDip<0.05)/it
	result[2,10]<-sum(single3DClassicSilv<0.05)/it
	result[2,11]<-sum(single3DPCADip<0.05)/it
	result[2,12]<-sum(single3DPCASilv<0.05)/it
	result[2,13]<-mean(single3DPrincurveConv)
	result[2,14]<-round(sum(single3DPrincurveDip<0.05)/it,4)
	result[2,15]<-round(sum(single3DPrincurveSilv<0.05)/it,4)
	result[3,1]<-sum(single10DDip<0.05)/it
	result[3,2]<-sum(single10DSilv<0.05)/it
	result[3,3]<-round(mean(single10DHop),4)
	result[3,4]<-round(sum(single10DHop<0.25)/it,4)
	result[3,5]<-round(sum(single10DHop<qbeta(0.025, nrow(singleClust10D)/10, nrow(singleClust10D)/10))/it,4)
	result[3,6]<-round(sum(single10DHop<qbeta(0.05, nrow(singleClust10D)/10, nrow(singleClust10D)/10))/it,4)
	result[3,7]<-round(sum(single10DHop>qbeta(0.95, nrow(singleClust10D)/10, nrow(singleClust10D)/10))/it,4)
	result[3,8]<-round(sum(single10DHop>qbeta(0.975, nrow(singleClust10D)/10, nrow(singleClust10D)/10))/it,4)
	result[3,9]<-sum(single10DClassicDip<0.05)/it	
	result[3,10]<-sum(single10DClassicSilv<0.05)/it
	result[3,11]<-sum(single10DPCADip<0.05)/it
	result[3,12]<-sum(single10DPCASilv<0.05)/it	
	result[3,13]<-mean(single10DPrincurveConv)
	result[3,14]<-round(sum(single10DPrincurveDip<0.05)/it,4)
	result[3,15]<-round(sum(single10DPrincurveSilv<0.05)/it,4)
	result[4,1]<-sum(singleClust50DDip<0.05)/it
	result[4,2]<-sum(singleClust50DSilv<0.05)/it
	result[4,3]<-round(mean(singleClust50DHop),4)
	result[4,4]<-round(sum(singleClust50DHop<0.25)/it,4)
	result[4,5]<-round(sum(singleClust50DHop<qbeta(0.025, nrow(singleClust50D)/10, nrow(singleClust50D)/10))/it,4)
	result[4,6]<-round(sum(singleClust50DHop<qbeta(0.05, nrow(singleClust50D)/10, nrow(singleClust50D)/10))/it,4)
	result[4,7]<-round(sum(singleClust50DHop>qbeta(0.95, nrow(singleClust50D)/10, nrow(singleClust50D)/10))/it,4)
	result[4,8]<-round(sum(singleClust50DHop>qbeta(0.975, nrow(singleClust50D)/10, nrow(singleClust50D)/10))/it,4)
	result[4,9]<-sum(singleClust50DClassicDip<0.05)/it
	result[4,10]<-sum(singleClust50DClassicSilv<0.05)/it
	result[4,11]<-sum(singleClust50DPCADip<0.05)/it
	result[4,12]<-sum(singleClust50DPCASilv<0.05)/it
	result[4,13]<-mean(singleClust50DPrincurveConv)
	result[4,14]<-round(sum(singleClust50DPrincurveDip<0.05)/it,4)
	result[4,15]<-round(sum(singleClust50DPrincurveSilv<0.05)/it,4)
	result[5,1]<-sum(singleOutlierDip<0.05)/it
	result[5,2]<-sum(singleOutlierSilv<0.05)/it
	result[5,3]<-round(mean(singleOutlierHop),4)
	result[5,4]<-round(sum(singleOutlierHop<0.25)/it,4)
	result[5,5]<-round(sum(singleOutlierHop<qbeta(0.025, nrow(singleClustOutlier)/10, nrow(singleClustOutlier)/10))/it,4)
	result[5,6]<-round(sum(singleOutlierHop<qbeta(0.05, nrow(singleClustOutlier)/10, nrow(singleClustOutlier)/10))/it,4)
	result[5,7]<-round(sum(singleOutlierHop>qbeta(0.95, nrow(singleClustOutlier)/10, nrow(singleClustOutlier)/10))/it,4)
	result[5,8]<-round(sum(singleOutlierHop>qbeta(0.975, nrow(singleClustOutlier)/10, nrow(singleClustOutlier)/10))/it,4)
	result[5,9]<-sum(singleOutlierClassicDip<0.05)/it
	result[5,10]<-sum(singleOutlierClassicSilv<0.05)/it
	result[5,11]<-sum(singleOutlierPCADip<0.05)/it
	result[5,12]<-sum(singleOutlierPCASilv<0.05)/it
	result[5,13]<-mean(singleOutlierPrincurveConv)
	result[5,14]<-round(sum(singleOutlierPrincurveDip<0.05)/it,4)
	result[5,15]<-round(sum(singleOutlierPrincurveSilv<0.05)/it,4)
	result[6,1]<-sum(largeOutlierDip<0.05)/it
	result[6,2]<-sum(largeOutlierSilv<0.05)/it
	result[6,3]<-round(mean(largeOutlierHop),4)
	result[6,4]<-round(sum(largeOutlierHop<0.25)/it,4)
	result[6,5]<-round(sum(largeOutlierHop<qbeta(0.025, nrow(largeClustOutlier)/10, nrow(largeClustOutlier)/10))/it,4)
	result[6,6]<-round(sum(largeOutlierHop<qbeta(0.05, nrow(largeClustOutlier)/10, nrow(largeClustOutlier)/10))/it,4)
	result[6,7]<-round(sum(largeOutlierHop>qbeta(0.95, nrow(largeClustOutlier)/10, nrow(largeClustOutlier)/10))/it,4)
	result[6,8]<-round(sum(largeOutlierHop>qbeta(0.975, nrow(largeClustOutlier)/10, nrow(largeClustOutlier)/10))/it,4)
	result[6,9]<-sum(largeOutlierClassicDip<0.05)/it
	result[6,10]<-sum(largeOutlierClassicSilv<0.05)/it
	result[6,11]<-sum(largeOutlierPCADip<0.05)/it
	result[6,12]<-sum(largeOutlierPCASilv<0.05)/it
	result[6,13]<-mean(largeOutlierPrincurveConv)
	result[6,14]<-round(sum(largeOutlierPrincurveDip<0.05)/it,4)
	result[6,15]<-round(sum(largeOutlierPrincurveSilv<0.05)/it,4)
	result[7,1]<-sum(single3OutliersDip<0.05)/it
	result[7,2]<-sum(single3OutliersSilv<0.05)/it
	result[7,3]<-round(mean(single3OutliersHop),4)
	result[7,4]<-round(sum(single3OutliersHop<0.25)/it,4)
	result[7,5]<-round(sum(single3OutliersHop<qbeta(0.025, nrow(singleClust3Outliers)/10, nrow(singleClust3Outliers)/10))/it,4)
	result[7,6]<-round(sum(single3OutliersHop<qbeta(0.05, nrow(singleClust3Outliers)/10, nrow(singleClust3Outliers)/10))/it,4)
	result[7,7]<-round(sum(single3OutliersHop>qbeta(0.95, nrow(singleClust3Outliers)/10, nrow(singleClust3Outliers)/10))/it,4)
	result[7,8]<-round(sum(single3OutliersHop>qbeta(0.975, nrow(singleClust3Outliers)/10, nrow(singleClust3Outliers)/10))/it,4)
	result[7,9]<-sum(single3OutliersClassicDip<0.05)/it
	result[7,10]<-sum(single3OutliersClassicSilv<0.05)/it
	result[7,11]<-sum(single3OutliersPCADip<0.05)/it
	result[7,12]<-sum(single3OutliersPCASilv<0.05)/it
	result[7,13]<-mean(single3OutliersPrincurveConv)
	result[7,14]<-round(sum(single3OutliersPrincurveDip<0.05)/it,4)
	result[7,15]<-round(sum(single3OutliersPrincurveSilv<0.05)/it,4)
	result[8,1]<-sum(oneTclust5dfDip<0.05)/it
	result[8,2]<-sum(oneTclust5dfSilv<0.05)/it
	result[8,3]<-round(mean(oneTclust5dfHop),4)
	result[8,4]<-round(sum(oneTclust5dfHop<0.25)/it,4)
	result[8,5]<-round(sum(oneTclust5dfHop<qbeta(0.025, nrow(oneTclust5df)/10, nrow(oneTclust5df)/10))/it,4)
	result[8,6]<-round(sum(oneTclust5dfHop<qbeta(0.05, nrow(oneTclust5df)/10, nrow(oneTclust5df)/10))/it,4)
	result[8,7]<-round(sum(oneTclust5dfHop>qbeta(0.95, nrow(oneTclust5df)/10, nrow(oneTclust5df)/10))/it,4)
	result[8,8]<-round(sum(oneTclust5dfHop>qbeta(0.975, nrow(oneTclust5df)/10, nrow(oneTclust5df)/10))/it,4)
	result[8,9]<-sum(oneTclust5dfClassicDip<0.05)/it
	result[8,10]<-sum(oneTclust5dfClassicSilv<0.05)/it
	result[8,11]<-sum(oneTclust5dfPCADip<0.05)/it
	result[8,12]<-sum(oneTclust5dfPCASilv<0.05)/it
	result[8,13]<-mean(oneTclust5dfPrincurveConv)
	result[8,14]<-round(sum(oneTclust5dfPrincurveDip<0.05)/it,4)
	result[8,15]<-round(sum(oneTclust5dfPrincurveSilv<0.05)/it,4)
	result[9,1]<-sum(oneTclust10dfDip<0.05)/it
	result[9,2]<-sum(oneTclust10dfSilv<0.05)/it
	result[9,3]<-round(mean(oneTclust10dfHop),4)
	result[9,4]<-round(sum(oneTclust10dfHop<0.25)/it,4)
	result[9,5]<-round(sum(oneTclust10dfHop<qbeta(0.025, nrow(oneTclust10df)/10, nrow(oneTclust10df)/10))/it,4)
	result[9,6]<-round(sum(oneTclust10dfHop<qbeta(0.05, nrow(oneTclust10df)/10, nrow(oneTclust10df)/10))/it,4)
	result[9,7]<-round(sum(oneTclust10dfHop>qbeta(0.95, nrow(oneTclust10df)/10, nrow(oneTclust10df)/10))/it,4)
	result[9,8]<-round(sum(oneTclust10dfHop>qbeta(0.975, nrow(oneTclust10df)/10, nrow(oneTclust10df)/10))/it,4)
	result[9,9]<-sum(oneTclust10dfClassicDip<0.05)/it
	result[9,10]<-sum(oneTclust10dfClassicSilv<0.05)/it
	result[9,11]<-sum(oneTclust10dfPCADip<0.05)/it
	result[9,12]<-sum(oneTclust10dfPCASilv<0.05)/it
	result[9,13]<-mean(oneTclust10dfPrincurveConv)
	result[9,14]<-round(sum(oneTclust10dfPrincurveDip<0.05)/it,4)
	result[9,15]<-round(sum(oneTclust10dfPrincurveSilv<0.05)/it,4)
	result[10,1]<-sum(oneTclust15dfDip<0.05)/it
	result[10,2]<-sum(oneTclust15dfSilv<0.05)/it
	result[10,3]<-round(mean(oneTclust15dfHop),4)
	result[10,4]<-round(sum(oneTclust15dfHop<0.25)/it,4)
	result[10,5]<-round(sum(oneTclust15dfHop<qbeta(0.025, nrow(oneTclust15df)/10, nrow(oneTclust15df)/10))/it,4)
	result[10,6]<-round(sum(oneTclust15dfHop<qbeta(0.05, nrow(oneTclust15df)/10, nrow(oneTclust15df)/10))/it,4)
	result[10,7]<-round(sum(oneTclust15dfHop>qbeta(0.95, nrow(oneTclust15df)/10, nrow(oneTclust15df)/10))/it,4)
	result[10,8]<-round(sum(oneTclust15dfHop>qbeta(0.975, nrow(oneTclust15df)/10, nrow(oneTclust15df)/10))/it,4)
	result[10,9]<-sum(oneTclust15dfClassicDip<0.05)/it
	result[10,10]<-sum(oneTclust15dfClassicSilv<0.05)/it
	result[10,11]<-sum(oneTclust15dfPCADip<0.05)/it
	result[10,12]<-sum(oneTclust15dfPCASilv<0.05)/it
	result[10,13]<-mean(oneTclust15dfPrincurveConv)
	result[10,14]<-round(sum(oneTclust15dfPrincurveDip<0.05)/it,4)
	result[10,15]<-round(sum(oneTclust15dfPrincurveSilv<0.05)/it,4)

	result[11,1]<-sum(twoSepDip<0.05)/it
	result[11,2]<-sum(twoSepSilv<0.05)/it
	result[11,3]<-round(mean(twoSepHop),4)
	result[11,4]<-round(sum(twoSepHop<0.25)/it,4)
	result[11,5]<-round(sum(twoSepHop<qbeta(0.025, nrow(twoSepClust)/10, nrow(twoSepClust)/10))/it,4)
	result[11,6]<-round(sum(twoSepHop<qbeta(0.05, nrow(twoSepClust)/10, nrow(twoSepClust)/10))/it,4)
	result[11,7]<-round(sum(twoSepHop>qbeta(0.95, nrow(twoSepClust)/10, nrow(twoSepClust)/10))/it,4)
	result[11,8]<-round(sum(twoSepHop>qbeta(0.975, nrow(twoSepClust)/10, nrow(twoSepClust)/10))/it,4)
	result[11,9]<-sum(twoSepClassicDip<0.05)/it
	result[11,10]<-sum(twoSepClassicSilv<0.05)/it
	result[11,11]<-sum(twoSepPCADip<0.05)/it
	result[11,12]<-sum(twoSepPCASilv<0.05)/it
	result[11,13]<-mean(twoSepPrincurveConv)
	result[11,14]<-round(sum(twoSepPrincurveDip<0.05)/it,4)
	result[11,15]<-round(sum(twoSepPrincurveSilv<0.05)/it,4)
	result[12,1]<-sum(threeSepDip<0.05)/it
	result[12,2]<-sum(threeSepSilv<0.05)/it
	result[12,3]<-round(mean(threeSepHop),4)
	result[12,4]<-round(sum(threeSepHop<0.25)/it,4)
	result[12,5]<-round(sum(threeSepHop<qbeta(0.025, nrow(threeSepClust)/10, nrow(threeSepClust)/10))/it,4)
	result[12,6]<-round(sum(threeSepHop<qbeta(0.05, nrow(threeSepClust)/10, nrow(threeSepClust)/10))/it,4)
	result[12,7]<-round(sum(threeSepHop>qbeta(0.95, nrow(threeSepClust)/10, nrow(threeSepClust)/10))/it,4)
	result[12,8]<-round(sum(threeSepHop>qbeta(0.975, nrow(threeSepClust)/10, nrow(threeSepClust)/10))/it,4)
	result[12,9]<-sum(threeSepClassicDip<0.05)/it
	result[12,10]<-sum(threeSepClassicSilv<0.05)/it
	result[12,11]<-sum(threeSepPCADip<0.05)/it
	result[12,12]<-sum(threeSepPCASilv<0.05)/it
	result[12,13]<-mean(threeSepPrincurveConv)
	result[12,14]<-round(sum(threeSepPrincurveDip<0.05)/it,4)
	result[12,15]<-round(sum(threeSepPrincurveSilv<0.05)/it,4)
	result[13,1]<-sum(threeSep3DDip<0.05)/it
	result[13,2]<-sum(threeSep3DSilv<0.05)/it
	result[13,3]<-round(mean(threeSep3DHop),4)
	result[13,4]<-round(sum(threeSep3DHop<0.25)/it,4)
	result[13,5]<-round(sum(threeSep3DHop<qbeta(0.025, nrow(threeSepClust3D)/10, nrow(threeSepClust3D)/10))/it,4)
	result[13,6]<-round(sum(threeSep3DHop<qbeta(0.05, nrow(threeSepClust3D)/10, nrow(threeSepClust3D)/10))/it,4)
	result[13,7]<-round(sum(threeSep3DHop>qbeta(0.95, nrow(threeSepClust3D)/10, nrow(threeSepClust3D)/10))/it,4)
	result[13,8]<-round(sum(threeSep3DHop>qbeta(0.975, nrow(threeSepClust3D)/10, nrow(threeSepClust3D)/10))/it,4)
	result[13,9]<-sum(threeSep3DClassicDip<0.05)/it
	result[13,10]<-sum(threeSep3DClassicSilv<0.05)/it
	result[13,11]<-sum(threeSep3DPCADip<0.05)/it
	result[13,12]<-sum(threeSep3DPCASilv<0.05)/it
	result[13,13]<-mean(threeSep3DPrincurveConv)
	result[13,14]<-round(sum(threeSep3DPrincurveDip<0.05)/it,4)
	result[13,15]<-round(sum(threeSep3DPrincurveSilv<0.05)/it,4)
	result[14,1]<-sum(threeCloseDip<0.05)/it
	result[14,2]<-sum(threeCloseSilv<0.05)/it
	result[14,3]<-round(mean(threeCloseHop),4)
	result[14,4]<-round(sum(threeCloseHop<0.25)/it,4)
	result[14,5]<-round(sum(threeCloseHop<qbeta(0.025, nrow(threeCloseClust)/10, nrow(threeCloseClust)/10))/it,4)
	result[14,6]<-round(sum(threeCloseHop<qbeta(0.05, nrow(threeCloseClust)/10, nrow(threeCloseClust)/10))/it,4)
	result[14,7]<-round(sum(threeCloseHop>qbeta(0.95, nrow(threeCloseClust)/10, nrow(threeCloseClust)/10))/it,4)
	result[14,8]<-round(sum(threeCloseHop>qbeta(0.975, nrow(threeCloseClust)/10, nrow(threeCloseClust)/10))/it,4)
	result[14,9]<-sum(threeCloseClassicDip<0.05)/it
	result[14,10]<-sum(threeCloseClassicSilv<0.05)/it
	result[14,11]<-sum(threeClosePCADip<0.05)/it
	result[14,12]<-sum(threeClosePCASilv<0.05)/it
	result[14,13]<-mean(threeClosePrincurveConv)
	result[14,14]<-round(sum(threeClosePrincurveDip<0.05)/it,4)
	result[14,15]<-round(sum(threeClosePrincurveSilv<0.05)/it,4)
	result[15,1]<-sum(threeNoisyDip<0.05)/it
	result[15,2]<-sum(threeNoisySilv<0.05)/it
	result[15,3]<-round(mean(threeNoisyHop),4)
	result[15,4]<-round(sum(threeNoisyHop<0.25)/it,4)
	result[15,5]<-round(sum(threeNoisyHop<qbeta(0.025, nrow(threeNoisyClust)/10, nrow(threeNoisyClust)/10))/it,4)
	result[15,6]<-round(sum(threeNoisyHop<qbeta(0.05, nrow(threeNoisyClust)/10, nrow(threeNoisyClust)/10))/it,4)
	result[15,7]<-round(sum(threeNoisyHop>qbeta(0.95, nrow(threeNoisyClust)/10, nrow(threeNoisyClust)/10))/it,4)
	result[15,8]<-round(sum(threeNoisyHop>qbeta(0.975, nrow(threeNoisyClust)/10, nrow(threeNoisyClust)/10))/it,4)
	result[15,9]<-sum(threeNoisyClassicDip<0.05)/it
	result[15,10]<-sum(threeNoisyClassicSilv<0.05)/it
	result[15,11]<-sum(threeNoisyPCADip<0.05)/it
	result[15,12]<-sum(threeNoisyPCASilv<0.05)/it
	result[15,13]<-mean(threeNoisyPrincurveConv)
	result[15,14]<-round(sum(threeNoisyPrincurveDip<0.05)/it,4)
	result[15,15]<-round(sum(threeNoisyPrincurveSilv<0.05)/it,4)
	result[16,1]<-sum(threeDiffDiameterDip<0.05)/it
	result[16,2]<-sum(threeDiffDiameterSilv<0.05)/it
	result[16,3]<-round(mean(threeDiffDiameterHop),4)
	result[16,4]<-round(sum(threeDiffDiameterHop<0.25)/it,4)
	result[16,5]<-round(sum(threeDiffDiameterHop<qbeta(0.025, nrow(threeClustDiffDiameter)/10, nrow(threeClustDiffDiameter)/10))/it,4)
	result[16,6]<-round(sum(threeDiffDiameterHop<qbeta(0.05, nrow(threeClustDiffDiameter)/10, nrow(threeClustDiffDiameter)/10))/it,4)
	result[16,7]<-round(sum(threeDiffDiameterHop>qbeta(0.95, nrow(threeClustDiffDiameter)/10, nrow(threeClustDiffDiameter)/10))/it,4)
	result[16,8]<-round(sum(threeDiffDiameterHop>qbeta(0.975, nrow(threeClustDiffDiameter)/10, nrow(threeClustDiffDiameter)/10))/it,4)
	result[16,9]<-sum(threeDiffDiameterClassicDip<0.05)/it
	result[16,10]<-sum(threeDiffDiameterClassicSilv<0.05)/it
	result[16,11]<-sum(threeDiffDiameterPCADip<0.05)/it
	result[16,12]<-sum(threeDiffDiameterPCASilv<0.05)/it
	result[16,13]<-mean(threeDiffDiameterPrincurveConv)
	result[16,14]<-round(sum(threeDiffDiameterPrincurveDip<0.05)/it,4)
	result[16,15]<-round(sum(threeDiffDiameterPrincurveSilv<0.05)/it,4)
	result[17,1]<-sum(threeDiffDensityDip<0.05)/it
	result[17,2]<-sum(threeDiffDensitySilv<0.05)/it
	result[17,3]<-round(mean(threeDiffDensityHop),4)
	result[17,4]<-round(sum(threeDiffDensityHop<0.25)/it,4)
	result[17,5]<-round(sum(threeDiffDensityHop<qbeta(0.025, nrow(threeClustDiffDensity)/10, nrow(threeClustDiffDensity)/10))/it,4)
	result[17,6]<-round(sum(threeDiffDensityHop<qbeta(0.05, nrow(threeClustDiffDensity)/10, nrow(threeClustDiffDensity)/10))/it,4)
	result[17,7]<-round(sum(threeDiffDensityHop>qbeta(0.95, nrow(threeClustDiffDensity)/10, nrow(threeClustDiffDensity)/10))/it,4)
	result[17,8]<-round(sum(threeDiffDensityHop>qbeta(0.975, nrow(threeClustDiffDensity)/10, nrow(threeClustDiffDensity)/10))/it,4)
	result[17,9]<-sum(threeDiffDensityClassicDip<0.05)/it
	result[17,10]<-sum(threeDiffDensityClassicSilv<0.05)/it
	result[17,11]<-sum(threeDiffDensityPCADip<0.05)/it
	result[17,12]<-sum(threeDiffDensityPCASilv<0.05)/it
	result[17,13]<-mean(threeDiffDensityPrincurveConv)
	result[17,14]<-round(sum(threeDiffDensityPrincurveDip<0.05)/it,4)
	result[17,15]<-round(sum(threeDiffDensityPrincurveSilv<0.05)/it,4)
	result[18,1]<-sum(twoSep10DDip<0.05)/it
	result[18,2]<-sum(twoSep10DSilv<0.05)/it
	result[18,3]<-round(mean(twoSep10DHop),4)
	result[18,4]<-round(sum(twoSep10DHop<0.25)/it,4)
	result[18,5]<-round(sum(twoSep10DHop<qbeta(0.025, nrow(twoSepClust10D)/10, nrow(twoSepClust10D)/10))/it,4)
	result[18,6]<-round(sum(twoSep10DHop<qbeta(0.05, nrow(twoSepClust10D)/10, nrow(twoSepClust10D)/10))/it,4)
	result[18,7]<-round(sum(twoSep10DHop>qbeta(0.95, nrow(twoSepClust10D)/10, nrow(twoSepClust10D)/10))/it,4)
	result[18,8]<-round(sum(twoSep10DHop>qbeta(0.975, nrow(twoSepClust10D)/10, nrow(twoSepClust10D)/10))/it,4)
	result[18,9]<-sum(twoSep10DClassicDip<0.05)/it
	result[18,10]<-sum(twoSep10DClassicSilv<0.05)/it
	result[18,11]<-sum(twoSep10DPCADip<0.05)/it
	result[18,12]<-sum(twoSep10DPCASilv<0.05)/it
	result[18,13]<-mean(twoSep10DPrincurveConv)
	result[18,14]<-round(sum(twoSep10DPrincurveDip<0.05)/it,4)
	result[18,15]<-round(sum(twoSep10DPrincurveSilv<0.05)/it,4)
	result[19,1]<-sum(fourSep10DDip<0.05)/it
	result[19,2]<-sum(fourSep10DSilv<0.05)/it
	result[19,3]<-round(mean(fourSep10DHop),4)
	result[19,4]<-round(sum(fourSep10DHop<0.25)/it,4)
	result[19,5]<-round(sum(fourSep10DHop<qbeta(0.025, nrow(fourSepClust10D)/10, nrow(fourSepClust10D)/10))/it,4)
	result[19,6]<-round(sum(fourSep10DHop<qbeta(0.05, nrow(fourSepClust10D)/10, nrow(fourSepClust10D)/10))/it,4)
	result[19,7]<-round(sum(fourSep10DHop>qbeta(0.95, nrow(fourSepClust10D)/10, nrow(fourSepClust10D)/10))/it,4)
	result[19,8]<-round(sum(fourSep10DHop>qbeta(0.975, nrow(fourSepClust10D)/10, nrow(fourSepClust10D)/10))/it,4)
	result[19,9]<-sum(fourSep10DClassicDip<0.05)/it
	result[19,10]<-sum(fourSep10DClassicSilv<0.05)/it
	result[19,11]<-sum(fourSep10DPCADip<0.05)/it
	result[19,12]<-sum(fourSep10DPCASilv<0.05)/it
	result[19,13]<-mean(fourSep10DPrincurveConv)
	result[19,14]<-round(sum(fourSep10DPrincurveDip<0.05)/it,4)
	result[19,15]<-round(sum(fourSep10DPrincurveSilv<0.05)/it,4)
	result[20,1]<-sum(tenSeparatedDip<0.05)/it
	result[20,2]<-sum(tenSeparatedSilv<0.05)/it
	result[20,3]<-round(mean(tenSeparatedHop),4)
	result[20,4]<-round(sum(tenSeparatedHop<0.25)/it,4)
	result[20,5]<-round(sum(tenSeparatedHop<qbeta(0.025, nrow(tenSeparatedClusters)/10, nrow(tenSeparatedClusters)/10))/it,4)
	result[20,6]<-round(sum(tenSeparatedHop<qbeta(0.05, nrow(tenSeparatedClusters)/10, nrow(tenSeparatedClusters)/10))/it,4)
	result[20,7]<-round(sum(tenSeparatedHop>qbeta(0.95, nrow(tenSeparatedClusters)/10, nrow(tenSeparatedClusters)/10))/it,4)
	result[20,8]<-round(sum(tenSeparatedHop>qbeta(0.975, nrow(tenSeparatedClusters)/10, nrow(tenSeparatedClusters)/10))/it,4)
	result[20,9]<-sum(tenSeparatedClassicDip<0.05)/it
	result[20,10]<-sum(tenSeparatedClassicSilv<0.05)/it
	result[20,11]<-sum(tenSeparatedPCADip<0.05)/it
	result[20,12]<-sum(tenSeparatedPCASilv<0.05)/it
	result[20,13]<-mean(tenSeparatedPrincurveConv)
	result[20,14]<-round(sum(tenSeparatedPrincurveDip<0.05)/it,4)
	result[20,15]<-round(sum(tenSeparatedPrincurveSilv<0.05)/it,4)
	result[21,1]<-sum(tenCloseDip<0.05)/it
	result[21,2]<-sum(tenCloseSilv<0.05)/it
	result[21,3]<-round(mean(tenCloseHop),4)
	result[21,4]<-round(sum(tenCloseHop<0.25)/it,4)
	result[21,5]<-round(sum(tenCloseHop<qbeta(0.025, nrow(tenCloseClusters)/10, nrow(tenCloseClusters)/10))/it,4)
	result[21,6]<-round(sum(tenCloseHop<qbeta(0.05, nrow(tenCloseClusters)/10, nrow(tenCloseClusters)/10))/it,4)
	result[21,7]<-round(sum(tenCloseHop>qbeta(0.95, nrow(tenCloseClusters)/10, nrow(tenCloseClusters)/10))/it,4)
	result[21,8]<-round(sum(tenCloseHop>qbeta(0.975, nrow(tenCloseClusters)/10, nrow(tenCloseClusters)/10))/it,4)
	result[21,9]<-sum(tenCloseClassicDip<0.05)/it
	result[21,10]<-sum(tenCloseClassicSilv<0.05)/it
	result[21,11]<-sum(tenClosePCADip<0.05)/it
	result[21,12]<-sum(tenClosePCASilv<0.05)/it
	result[21,13]<-mean(tenClosePrincurveConv)
	result[21,14]<-round(sum(tenClosePrincurveDip<0.05)/it,4)
	result[21,15]<-round(sum(tenClosePrincurveSilv<0.05)/it,4)
	result[22,1]<-sum(sevenCloseDenseDip<0.05)/it
	result[22,2]<-sum(sevenCloseDenseSilv<0.05)/it
	result[22,3]<-round(mean(sevenCloseDenseHop),4)
	result[22,4]<-round(sum(sevenCloseDenseHop<0.25)/it,4)
	result[22,5]<-round(sum(sevenCloseDenseHop<qbeta(0.025, nrow(sevenCloseDenseClusters)/10, nrow(sevenCloseDenseClusters)/10))/it,4)
	result[22,6]<-round(sum(sevenCloseDenseHop<qbeta(0.05, nrow(sevenCloseDenseClusters)/10, nrow(sevenCloseDenseClusters)/10))/it,4)
	result[22,7]<-round(sum(sevenCloseDenseHop>qbeta(0.95, nrow(sevenCloseDenseClusters)/10, nrow(sevenCloseDenseClusters)/10))/it,4)
	result[22,8]<-round(sum(sevenCloseDenseHop>qbeta(0.975, nrow(sevenCloseDenseClusters)/10, nrow(sevenCloseDenseClusters)/10))/it,4)
	result[22,9]<-sum(sevenCloseDenseClassicDip<0.05)/it
	result[22,10]<-sum(sevenCloseDenseClassicSilv<0.05)/it
	result[22,11]<-sum(sevenCloseDensePCADip<0.05)/it
	result[22,12]<-sum(sevenCloseDensePCASilv<0.05)/it
	result[22,13]<-mean(sevenCloseDensePrincurveConv)
	result[22,14]<-round(sum(sevenCloseDensePrincurveDip<0.05)/it,4)
	result[22,15]<-round(sum(sevenCloseDensePrincurveSilv<0.05)/it,4)
	result[23,1]<-sum(sevenCloseDip<0.05)/it
	result[23,2]<-sum(sevenCloseSilv<0.05)/it
	result[23,3]<-round(mean(sevenCloseHop),4)
	result[23,4]<-round(sum(sevenCloseHop<0.25)/it,4)
	result[23,5]<-round(sum(sevenCloseHop<qbeta(0.025, nrow(sevenCloseClusters)/10, nrow(sevenCloseClusters)/10))/it,4)
	result[23,6]<-round(sum(sevenCloseHop<qbeta(0.05, nrow(sevenCloseClusters)/10, nrow(sevenCloseClusters)/10))/it,4)
	result[23,7]<-round(sum(sevenCloseHop>qbeta(0.95, nrow(sevenCloseClusters)/10, nrow(sevenCloseClusters)/10))/it,4)
	result[23,8]<-round(sum(sevenCloseHop>qbeta(0.975, nrow(sevenCloseClusters)/10, nrow(sevenCloseClusters)/10))/it,4)
	result[23,9]<-sum(sevenCloseClassicDip<0.05)/it
	result[23,10]<-sum(sevenCloseClassicSilv<0.05)/it
	result[23,11]<-sum(sevenClosePCADip<0.05)/it
	result[23,12]<-sum(sevenClosePCASilv<0.05)/it
	result[23,13]<-mean(sevenClosePrincurveConv)
	result[23,14]<-round(sum(sevenClosePrincurveDip<0.05)/it,4)
	result[23,15]<-round(sum(sevenClosePrincurveSilv<0.05)/it,4)
	result[24,1]<-sum(fiveCloseDip<0.05)/it
	result[24,2]<-sum(fiveCloseSilv<0.05)/it
	result[24,3]<-round(mean(fiveCloseHop),4)
	result[24,4]<-round(sum(fiveCloseHop<0.25)/it,4)
	result[24,5]<-round(sum(fiveCloseHop<qbeta(0.025, nrow(fiveCloseClusters)/10, nrow(fiveCloseClusters)/10))/it,4)
	result[24,6]<-round(sum(fiveCloseHop<qbeta(0.05, nrow(fiveCloseClusters)/10, nrow(fiveCloseClusters)/10))/it,4)
	result[24,7]<-round(sum(fiveCloseHop>qbeta(0.95, nrow(fiveCloseClusters)/10, nrow(fiveCloseClusters)/10))/it,4)
	result[24,8]<-round(sum(fiveCloseHop>qbeta(0.975, nrow(fiveCloseClusters)/10, nrow(fiveCloseClusters)/10))/it,4)
	result[24,9]<-sum(fiveCloseClassicDip<0.05)/it
	result[24,10]<-sum(fiveCloseClassicSilv<0.05)/it
	result[24,11]<-sum(fiveClosePCADip<0.05)/it
	result[24,12]<-sum(fiveClosePCASilv<0.05)/it
	result[24,13]<-mean(fiveClosePrincurveConv)
	result[24,14]<-round(sum(fiveClosePrincurveDip<0.05)/it,4)
	result[24,15]<-round(sum(fiveClosePrincurveSilv<0.05)/it,4)
	result[25,1]<-sum(parallelLinesDip<0.05)/it
	result[25,2]<-sum(parallelLinesSilv<0.05)/it
	result[25,3]<-round(mean(parallelLinesHop),4)
	result[25,4]<-round(sum(parallelLinesHop<0.25)/it,4)
	result[25,5]<-round(sum(parallelLinesHop<qbeta(0.025, nrow(twoSepLines)/10, nrow(twoSepLines)/10))/it,4)
	result[25,6]<-round(sum(parallelLinesHop<qbeta(0.05, nrow(twoSepLines)/10, nrow(twoSepLines)/10))/it,4)
	result[25,7]<-round(sum(parallelLinesHop>qbeta(0.95, nrow(twoSepLines)/10, nrow(twoSepLines)/10))/it,4)
	result[25,8]<-round(sum(parallelLinesHop>qbeta(0.975, nrow(twoSepLines)/10, nrow(twoSepLines)/10))/it,4)
	result[25,9]<-sum(parallelLinesClassicDip<0.05)/it
	result[25,10]<-sum(parallelLinesClassicSilv<0.05)/it	
	result[25,11]<-sum(parallelLinesPCADip<0.05)/it
	result[25,12]<-sum(parallelLinesPCASilv<0.05)/it
	result[25,13]<-mean(parallelLinesPrincurveConv)
	result[25,14]<-round(sum(parallelLinesPrincurveDip<0.05)/it,4)
	result[25,15]<-round(sum(parallelLinesPrincurveSilv<0.05)/it,4)
	result[26,1]<-sum(concCirclesDip<0.05)/it
	result[26,2]<-sum(concCirclesSilv<0.05)/it
	result[26,3]<-round(mean(concCirclesHop),4)
	result[26,4]<-round(sum(concCirclesHop<0.25)/it,4)
	result[26,5]<-round(sum(concCirclesHop<qbeta(0.025, nrow(concCircles)/10, nrow(concCircles)/10))/it,4)
	result[26,6]<-round(sum(concCirclesHop<qbeta(0.05, nrow(concCircles)/10, nrow(concCircles)/10))/it,4)
	result[26,7]<-round(sum(concCirclesHop>qbeta(0.95, nrow(concCircles)/10, nrow(concCircles)/10))/it,4)
	result[26,8]<-round(sum(concCirclesHop>qbeta(0.975, nrow(concCircles)/10, nrow(concCircles)/10))/it,4)
	result[26,9]<-sum(concCirclesClassicDip<0.05)/it
	result[26,10]<-sum(concCirclesClassicSilv<0.05)/it
	result[26,11]<-sum(concCirclesPCADip<0.05)/it
	result[26,12]<-sum(concCirclesPCASilv<0.05)/it
	result[26,13]<-mean(concCirclesPrincurveConv)
	result[26,14]<-round(sum(concCirclesPrincurveDip<0.05)/it,4)
	result[26,15]<-round(sum(concCirclesPrincurveSilv<0.05)/it,4)
	result[27,1]<-sum(threeConcCirclesDip<0.05)/it
	result[27,2]<-sum(threeConcCirclesSilv<0.05)/it
	result[27,3]<-round(mean(threeConcCirclesHop),4)
	result[27,4]<-round(sum(threeConcCirclesHop<0.25)/it,4)
	result[27,5]<-round(sum(threeConcCirclesHop<qbeta(0.025, nrow(threeConcCircles)/10, nrow(threeConcCircles)/10))/it,4)
	result[27,6]<-round(sum(threeConcCirclesHop<qbeta(0.05, nrow(threeConcCircles)/10, nrow(threeConcCircles)/10))/it,4)
	result[27,7]<-round(sum(threeConcCirclesHop>qbeta(0.95, nrow(threeConcCircles)/10, nrow(threeConcCircles)/10))/it,4)
	result[27,8]<-round(sum(threeConcCirclesHop>qbeta(0.975, nrow(threeConcCircles)/10, nrow(threeConcCircles)/10))/it,4)
	result[27,9]<-sum(threeConcCirclesClassicDip<0.05)/it
	result[27,10]<-sum(threeConcCirclesClassicSilv<0.05)/it
	result[27,11]<-sum(threeConcCirclesPCADip<0.05)/it
	result[27,12]<-sum(threeConcCirclesPCASilv<0.05)/it
	result[27,13]<-mean(threeConcCirclesPrincurveConv)
	result[27,14]<-round(sum(threeConcCirclesPrincurveDip<0.05)/it,4)
	result[27,15]<-round(sum(threeConcCirclesPrincurveSilv<0.05)/it,4)
	result[28,1]<-sum(fiveConcCircleDip<0.05)/it
	result[28,2]<-sum(fiveConcCircleSilv<0.05)/it
	result[28,3]<-round(mean(fiveConcCircleHop),4)
	result[28,4]<-round(sum(fiveConcCircleHop<0.25)/it,4)
	result[28,5]<-round(sum(fiveConcCircleHop<qbeta(0.025, nrow(fiveConcCircles)/10, nrow(fiveConcCircles)/10))/it,4)
	result[28,6]<-round(sum(fiveConcCircleHop<qbeta(0.05, nrow(fiveConcCircles)/10, nrow(fiveConcCircles)/10))/it,4)
	result[28,7]<-round(sum(fiveConcCircleHop>qbeta(0.95, nrow(fiveConcCircles)/10, nrow(fiveConcCircles)/10))/it,4)
	result[28,8]<-round(sum(fiveConcCircleHop>qbeta(0.975, nrow(fiveConcCircles)/10, nrow(fiveConcCircles)/10))/it,4)
	result[28,9]<-sum(fiveConcCircleClassicDip<0.05)/it
	result[28,10]<-sum(fiveConcCircleClassicSilv<0.05)/it
	result[28,11]<-sum(fiveConcCirclePCADip<0.05)/it
	result[28,12]<-sum(fiveConcCirclePCASilv<0.05)/it
	result[28,13]<-mean(fiveConcCirclePrincurveConv)
	result[28,14]<-round(sum(fiveConcCirclePrincurveDip<0.05)/it,4)
	result[28,15]<-round(sum(fiveConcCirclePrincurveSilv<0.05)/it,4)
	result[29,1]<-sum(twoTclust10dfDip<0.05)/it
	result[29,2]<-sum(twoTclust10dfSilv<0.05)/it
	result[29,3]<-round(mean(twoTclust10dfHop),4)
	result[29,4]<-round(sum(twoTclust10dfHop<0.25)/it,4)
	result[29,5]<-round(sum(twoTclust10dfHop<qbeta(0.025, nrow(twoTclust10df)/10, nrow(twoTclust10df)/10))/it,4)
	result[29,6]<-round(sum(twoTclust10dfHop<qbeta(0.05, nrow(twoTclust10df)/10, nrow(twoTclust10df)/10))/it,4)
	result[29,7]<-round(sum(twoTclust10dfHop>qbeta(0.95, nrow(twoTclust10df)/10, nrow(twoTclust10df)/10))/it,4)
	result[29,8]<-round(sum(twoTclust10dfHop>qbeta(0.975, nrow(twoTclust10df)/10, nrow(twoTclust10df)/10))/it,4)
	result[29,9]<-sum(twoTclust10dfClassicDip<0.05)/it
	result[29,10]<-sum(twoTclust10dfClassicSilv<0.05)/it
	result[29,11]<-sum(twoTclust10dfPCADip<0.05)/it
	result[29,12]<-sum(twoTclust10dfPCASilv<0.05)/it
	result[29,13]<-mean(twoTclust10dfPrincurveConv)
	result[29,14]<-round(sum(twoTclust10dfPrincurveDip<0.05)/it,4)
	result[29,15]<-round(sum(twoTclust10dfPrincurveSilv<0.05)/it,4)
	result[30,1]<-sum(twoTclust5dfDip<0.05)/it
	result[30,2]<-sum(twoTclust5dfSilv<0.05)/it
	result[30,3]<-round(mean(twoTclust5dfHop),4)
	result[30,4]<-round(sum(twoTclust5dfHop<0.25)/it,4)
	result[30,5]<-round(sum(twoTclust5dfHop<qbeta(0.025, nrow(twoTclust5df)/10, nrow(twoTclust5df)/10))/it,4)
	result[30,6]<-round(sum(twoTclust5dfHop<qbeta(0.05, nrow(twoTclust5df)/10, nrow(twoTclust5df)/10))/it,4)
	result[30,7]<-round(sum(twoTclust5dfHop>qbeta(0.95, nrow(twoTclust5df)/10, nrow(twoTclust5df)/10))/it,4)
	result[30,8]<-round(sum(twoTclust5dfHop>qbeta(0.975, nrow(twoTclust5df)/10, nrow(twoTclust5df)/10))/it,4)
	result[30,9]<-sum(twoTclust5dfClassicDip<0.05)/it
	result[30,10]<-sum(twoTclust5dfClassicSilv<0.05)/it
	result[30,11]<-sum(twoTclust5dfPCADip<0.05)/it
	result[30,12]<-sum(twoTclust5dfPCASilv<0.05)/it
	result[30,13]<-mean(twoTclust5dfPrincurveConv)
	result[30,14]<-round(sum(twoTclust5dfPrincurveDip<0.05)/it,4)
	result[30,15]<-round(sum(twoTclust5dfPrincurveSilv<0.05)/it,4)
	result[31,1]<-sum(twoTclust15dfDip<0.05)/it
	result[31,2]<-sum(twoTclust15dfSilv<0.05)/it
	result[31,3]<-round(mean(twoTclust15dfHop),4)
	result[31,4]<-round(sum(twoTclust15dfHop<0.25)/it,4)
	result[31,5]<-round(sum(twoTclust15dfHop<qbeta(0.025, nrow(twoTclust15df)/10, nrow(twoTclust15df)/10))/it,4)
	result[31,6]<-round(sum(twoTclust15dfHop<qbeta(0.05, nrow(twoTclust15df)/10, nrow(twoTclust15df)/10))/it,4)
	result[31,7]<-round(sum(twoTclust15dfHop>qbeta(0.95, nrow(twoTclust15df)/10, nrow(twoTclust15df)/10))/it,4)
	result[31,8]<-round(sum(twoTclust15dfHop>qbeta(0.975, nrow(twoTclust15df)/10, nrow(twoTclust15df)/10))/it,4)
	result[31,9]<-sum(twoTclust15dfClassicDip<0.05)/it
	result[31,10]<-sum(twoTclust15dfClassicSilv<0.05)/it
	result[31,11]<-sum(twoTclust15dfPCADip<0.05)/it
	result[31,12]<-sum(twoTclust15dfPCASilv<0.05)/it
	result[31,13]<-mean(twoTclust15dfPrincurveConv)
	result[31,14]<-round(sum(twoTclust15dfPrincurveDip<0.05)/it,4)
	result[31,15]<-round(sum(twoTclust15dfPrincurveSilv<0.05)/it,4)
	result[32,1]<-sum(twoOverlapClust50DDip<0.05)/it
	result[32,2]<-sum(twoOverlapClust50DSilv<0.05)/it
	result[32,3]<-round(mean(twoOverlapClust50DHop),4)
	result[32,4]<-round(sum(twoOverlapClust50DHop<0.25)/it,4)
	result[32,5]<-round(sum(twoOverlapClust50DHop<qbeta(0.025, nrow(twoOverlapClust50D)/10, nrow(twoOverlapClust50D)/10))/it,4)
	result[32,6]<-round(sum(twoOverlapClust50DHop<qbeta(0.05, nrow(twoOverlapClust50D)/10, nrow(twoOverlapClust50D)/10))/it,4)
	result[32,7]<-round(sum(twoOverlapClust50DHop>qbeta(0.95, nrow(twoOverlapClust50D)/10, nrow(twoOverlapClust50D)/10))/it,4)
	result[32,8]<-round(sum(twoOverlapClust50DHop>qbeta(0.975, nrow(twoOverlapClust50D)/10, nrow(twoOverlapClust50D)/10))/it,4)
	result[32,9]<-sum(twoOverlapClust50DClassicDip<0.05)/it
	result[32,10]<-sum(twoOverlapClust50DClassicSilv<0.05)/it
	result[32,11]<-sum(twoOverlapClust50DPCADip<0.05)/it
	result[32,12]<-sum(twoOverlapClust50DPCASilv<0.05)/it
	result[32,13]<-mean(twoOverlapClust50DPrincurveConv)
	result[32,14]<-round(sum(twoOverlapClust50DPrincurveDip<0.05)/it,4)
	result[32,15]<-round(sum(twoOverlapClust50DPrincurveSilv<0.05)/it,4)
	result[33,1]<-sum(twoCloseClust50DDip<0.05)/it
	result[33,2]<-sum(twoCloseClust50DSilv<0.05)/it
	result[33,3]<-round(mean(twoCloseClust50DHop),4)
	result[33,4]<-round(sum(twoCloseClust50DHop<0.25)/it,4)
	result[33,5]<-round(sum(twoCloseClust50DHop<qbeta(0.025, nrow(twoCloseClust50D)/10, nrow(twoCloseClust50D)/10))/it,4)
	result[33,6]<-round(sum(twoCloseClust50DHop<qbeta(0.05, nrow(twoCloseClust50D)/10, nrow(twoCloseClust50D)/10))/it,4)
	result[33,7]<-round(sum(twoCloseClust50DHop>qbeta(0.95, nrow(twoCloseClust50D)/10, nrow(twoCloseClust50D)/10))/it,4)
	result[33,8]<-round(sum(twoCloseClust50DHop>qbeta(0.975, nrow(twoCloseClust50D)/10, nrow(twoCloseClust50D)/10))/it,4)
	result[33,9]<-sum(twoCloseClust50DClassicDip<0.05)/it
	result[33,10]<-sum(twoCloseClust50DClassicSilv<0.05)/it
	result[33,11]<-sum(twoCloseClust50DPCADip<0.05)/it
	result[33,12]<-sum(twoCloseClust50DPCASilv<0.05)/it
	result[33,13]<-mean(twoCloseClust50DPrincurveConv)
	result[33,14]<-round(sum(twoCloseClust50DPrincurveDip<0.05)/it,4)
	result[33,15]<-round(sum(twoCloseClust50DPrincurveSilv<0.05)/it,4)


	result[34,1]<-sum(naomiDip<0.05)/it
	result[34,2]<-sum(naomiSilv<0.05)/it
	result[34,3]<-round(mean(naomiHop),4)
	result[34,4]<-round(sum(naomiHop<0.25)/it,4)
	result[34,5]<-round(sum(naomiHop<qbeta(0.025, nrow(naomiClust)/10, nrow(naomiClust)/10))/it,4)
	result[34,6]<-round(sum(naomiHop<qbeta(0.05, nrow(naomiClust)/10, nrow(naomiClust)/10))/it,4)
	result[34,7]<-round(sum(naomiHop>qbeta(0.95, nrow(naomiClust)/10, nrow(naomiClust)/10))/it,4)
	result[34,8]<-round(sum(naomiHop>qbeta(0.975, nrow(naomiClust)/10, nrow(naomiClust)/10))/it,4)
	result[34,9]<-sum(naomiClassicDip<0.05)/it
	result[34,10]<-sum(naomiClassicSilv<0.05)/it
	result[34,11]<-sum(naomiPCADip<0.05)/it
	result[34,12]<-sum(naomiPCASilv<0.05)/it
	result[34,13]<-mean(naomiPrincurveConv)
	result[34,14]<-round(sum(naomiPrincurveDip<0.05)/it,4)
	result[34,15]<-round(sum(naomiPrincurveSilv<0.05)/it,4)


	colnames(result)<-c("Dip","Silv.","Hop(Mean)","Hop<0.25","Hop<qbeta.025","Hop<qbeta.05","Hop>qbeta.95","Hop>qbeta.975",
					"Classic Dip","Classic Silv.","PCA Dip", "PCA Silv.","Pcurv.Conv","Pcurv.Dip","Pcurv.Silv")
	rownames(result)<-c("single Cluster","single Cluster 3D","single Cluster 10D","single Cluster 50D",
					"singleClustOutlier","largeClustOutlier","singleClust3Outliers","singleTcluster5df",
					"singleTcluster10df","singleTcluster15df",
		# multi-clusters:
					"twoSepClust","threeSepClust","threeSepClust3D","threeCloseClust","threeNoisyClust",
					"threeClustDiffDiameter","threeClustDiffDensity","twoSepClust10D","fourSepClust10D",
					"tenSeparatedClusters","tenCloseClusters","sevenCloseDenseClusters","sevenCloseClusters",
					"fiveCloseClusters","ParallelLines","twoConcentricCircles","threeConcentricCircles",
					"fiveConcentricCircles","twoTclusters5df","twoTclusters10df","twoTclusters15df",
					"twoOverlappingClust50D","twoCloseClust50D",
					"Naomis cluster")

	print(timer)
	cat("\n")
	print(result)
	
	write.table(timer,sep="&",file="simulationtimer.txt")
	write.table(result,sep="&",file="simulationresult.txt")
}


# cluster(n,x,y,sd)

threeCLOSERSeparatedClusters<-function(){
	c1<-cluster(50,35,40,2)
	c2<-cluster(50,65,40,2)
	c3<-cluster(50,50,60,2)
	g<-rbind(c1,c2,c3)
	return(g)
}

threeCLOSERSeparatedClusters3D<-function(){
	c1<-cluster3D(50,20,20,20,2)
	c2<-cluster3D(50,40,40,40,2)
	c3<-cluster3D(50,60,60,60,2)
	g<-rbind(c1,c2,c3)
	return(g)
}

threeCLOSERClustersVaryingDensity<-function(){
	c1<-cluster(100,35,40,2)
	c2<-cluster(66,65,40,2)
	c3<-cluster(33,50,60,2)
	g<-rbind(c1,c2,c3)
	return(g)	
}

threeCLOSERNoisyClusters<-function(){
	c1<-cluster(50,30,40,2)
	c2<-cluster(50,70,40,2)
	c3<-cluster(50,50,80,2)
	n<-noise(80,50,50)
	g<-rbind(c1,c2,c3,n)
	return(g)
}

threeCLOSERClustersVaryingDiameter<-function(){
	c1<-cluster(50,30,40,1)
	c2<-cluster(50,70,40,3)
	c3<-cluster(50,50,80,5)
	g<-rbind(c1,c2,c3)
	return(g)
}

twoCLOSERSeparatedClusters10D<-function(){
	c1<-cluster10D(50,1,2)
	c2<-cluster10D(50,2,2)
	g<-rbind(c1,c2)
	return(g)
}

fourCLOSERSeparatedClusters10D<-function(){
	c1<-cluster10D(50,1,2)
	c2<-cluster10D(50,2,2)
	c3<-cluster10D(50,3,2)
	c4<-cluster10D(50,4,2)
	g<-rbind(c1,c2,c3,c4)
	return(g)
}

threeRowClusters<- function(){
	c1<-cluster(50,20,50,2)
	c2<-cluster(50,30,50,2)
	c3<-cluster(50,40,50,2)
	g<-rbind(c1,c2,c3)
	return(g)
}

fiveRowClusters<-function(){
	c1<-cluster(50,20,50,2)
	c2<-cluster(50,30,50,2)
	c3<-cluster(50,40,50,2)
	c4<-cluster(50,50,50,2)
	c5<-cluster(50,60,50,2)
	g<-rbind(c1,c2,c3,c4,c5)
	return(g)
}

sevenRowClusters<-function(){
	c1<-cluster(50,20,50,2)
	c2<-cluster(50,30,50,2)
	c3<-cluster(50,40,50,2)
	c4<-cluster(50,50,50,2)
	c5<-cluster(50,60,50,2)
	c6<-cluster(50,70,50,2)
	c7<-cluster(50,80,50,2)
	g<-rbind(c1,c2,c3,c4,c5,c6,c7)
	return(g)
}


oneRun<-function(it){
	clock = 0

	Silv=rep(NA,it)
	Dip=rep(NA,it)
	Hop=rep(NA,it)
	ClassicDip=rep(NA,it)
	ClassicSilv=rep(NA,it)
	PCASilv=rep(NA,it)
	PCADip=rep(NA,it)
	PrincurveConv=rep(NA,it)
	PrincurveDip=rep(NA,it)
	PrincurveSilv=rep(NA,it)
	PrincurveDipClock=0
	PrincurveSilvClock=0
	SilvClock=0
	DipClock=0
	HopClock=0
	ClassicDipClock=0
	ClassicSilvClock=0
	PCASilvClock=0
	PCADipClock=0


	for(i in 1:it){
		loopTimer <- proc.time()		#starts loop stopwatch
	
	#create
		#Circle(50)
		#Line(100,25, 20)
		#Line+Circle
			#rbind(Line(50,5,2)-50, Circle(100)*3)


			# *********************************************
		Clust<-rbind(Line(50,5,2)-50, Circle(100)*3)   #get data here
			# *********************************************


	#dist
		checkpoint<-Sys.time()
		Silv[i]<-silverman.test(dist(Clust),1,adjust=TRUE)@p_value
		SilvClock = SilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		Dip[i]<-dip.test(dist(Clust))$p.value
		DipClock = DipClock + as.numeric(Sys.time()-checkpoint)

	#Hop
		checkpoint<-Sys.time()
		Hop[i]<-hopkins(as.data.frame(Clust), n = nrow(Clust)/10)$H
		HopClock = HopClock + as.numeric(Sys.time()-checkpoint)

	#Classic
		checkpoint<-Sys.time()
		ClassicDip[i]<-dip.test(Clust)$p.value
		ClassicDipClock = ClassicDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		ClassicSilv[i]<-silverman.test(Clust,1,adjust=TRUE)@p_value
		ClassicSilvClock = ClassicSilvClock + as.numeric(Sys.time()-checkpoint)
			
	#PCA	
		checkpoint<-Sys.time()
		PCASilv[i]<-silverman.test(prcomp(Clust)$x[,1],1,adjust=TRUE)@p_value
		PCASilvClock = PCASilvClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		PCADip[i]<-dip.test(prcomp(Clust)$x[,1])$p.value
		PCADipClock = PCADipClock + as.numeric(Sys.time()-checkpoint)


	#PRINCIPAL CURVE
		pcit=100

		checkpoint<-Sys.time()
		Princurve<-principal.curve(Clust, maxit=pcit)
		PrincurveDipClock = PrincurveDipClock + as.numeric(Sys.time()-checkpoint)
		PrincurveSilvClock = PrincurveSilvClock + as.numeric(Sys.time()-checkpoint)

	#Dip and Silvcheck
		checkpoint<-Sys.time()
		PrincurveDip[i]<-dip.test(Princurve$s[,1])$p.value
		PrincurveDipClock = PrincurveDipClock + as.numeric(Sys.time()-checkpoint)

		checkpoint<-Sys.time()
		PrincurveSilv[i]<-silverman.test(Princurve$s[,1],1,adjust=TRUE)@p_value
		PrincurveSilvClock = PrincurveSilvClock + as.numeric(Sys.time()-checkpoint)
	
	#find convergence
		PrincurveConv[i]<-Princurve$converged	

	#feedback
		loopClock<-as.numeric(proc.time()-loopTimer)[3]		#stop loopstopwatch, and extract elapsed time
		clock=clock+loopClock						#add to total time passed.
		clockRemain=((3*(clock/i)+(loopClock))/4)*(it-i)		#predict future based on average of current simulation speed vs history

		cat(" Finished Generation: ", i, "/", it)
		printTime(prefix=" in ", loopClock)
		printTime(prefix="Total Time Elapsed: ", clock)  
		printTime(prefix="Estimated Time Remaining: ",clockRemain,suffix="\n")	
		flush.console()
		Sys.sleep(0.001)
	}

	timer<-matrix(,1,9)
	result<-matrix(,1,15)

	timer[1,1]<-DipClock/it
	timer[1,2]<-SilvClock/it
	timer[1,3]<-round(HopClock/it,4)
	timer[1,4]<-ClassicDipClock/it
	timer[1,5]<-ClassicSilvClock/it
	timer[1,6]<-PCADipClock/it
	timer[1,7]<-PCASilvClock/it
	timer[1,8]<-round(PrincurveDipClock/it,4)
	timer[1,9]<-round(PrincurveSilvClock/it,4)


	colnames(timer)<-c("Dip","Silv.","Hop","Classic Dip","Classic Silv.","PCA Dip", "PCA Silv.","Pcurv.Dip","Pcurv.Silv")
	rownames(timer)<-c("oneRun Cluster")


	result[1,1]<-sum(Dip<0.05)/it
	result[1,2]<-sum(Silv<0.05)/it
	result[1,3]<-round(mean(Hop),4)
	result[1,4]<-round(sum(Hop<0.25)/it,4)
	result[1,5]<-round(sum(Hop<qbeta(0.025, nrow(Clust)/10, nrow(Clust)/10))/it,4)
	result[1,6]<-round(sum(Hop<qbeta(0.05, nrow(Clust)/10, nrow(Clust)/10))/it,4)
	result[1,7]<-round(sum(Hop>qbeta(0.95, nrow(Clust)/10, nrow(Clust)/10))/it,4)
	result[1,8]<-round(sum(Hop>qbeta(0.975, nrow(Clust)/10, nrow(Clust)/10))/it,4)
	result[1,9]<-sum(ClassicDip<0.05)/it
	result[1,10]<-sum(ClassicSilv<0.05)/it
	result[1,11]<-sum(PCADip<0.05)/it
	result[1,12]<-sum(PCASilv<0.05)/it
	result[1,13]<-mean(PrincurveConv)
	result[1,14]<-round(sum(PrincurveDip<0.05)/it,4)
	result[1,15]<-round(sum(PrincurveSilv<0.05)/it,4)


	colnames(result)<-c("Dip","Silv.","Hop(Mean)","Hop<0.25","Hop<qbeta.025","Hop<qbeta.05","Hop>qbeta.95","Hop>qbeta.975",
					"Classic Dip","Classic Silv.","PCA Dip", "PCA Silv.","Pcurv.Conv","Pcurv.Dip","Pcurv.Silv")
	rownames(result)<-c("oneRun Cluster")

	print(timer)
	cat("\n")
	print(result)
	
	write.table(timer,sep="&",file="oneRunTimer.txt")
	write.table(result,sep="&",file="oneRunResult.txt")

}


quickRun<-function(it, hopN = 50){
	clock = 0	

	threeSepSilv=rep(NA,it)
	threeSepDip=rep(NA,it)
	threeSepHop=rep(NA,it)
	threeSepClassicSilv=rep(NA,it)
	threeSepClassicDip=rep(NA,it)
	threeSepPCASilv=rep(NA,it)
	threeSepPCADip=rep(NA,it)	
	threeSep3DSilv=rep(NA,it)
	threeSep3DDip=rep(NA,it)
	threeSep3DHop=rep(NA,it)
	threeSep3DClassicSilv=rep(NA,it)
	threeSep3DClassicDip=rep(NA,it)
	threeSep3DPCASilv=rep(NA,it)
	threeSep3DPCADip=rep(NA,it)
	threeNoisySilv=rep(NA,it)
	threeNoisyDip=rep(NA,it)
	threeNoisyHop=rep(NA,it)
	threeNoisyClassicSilv=rep(NA,it)
	threeNoisyClassicDip=rep(NA,it)
	threeNoisyPCASilv=rep(NA,it)
	threeNoisyPCADip=rep(NA,it)
	threeDiffDiameterSilv=rep(NA,it)
	threeDiffDiameterDip=rep(NA,it)
	threeDiffDiameterHop=rep(NA,it)
	threeDiffDiameterClassicSilv=rep(NA,it)
	threeDiffDiameterClassicDip=rep(NA,it)
	threeDiffDiameterPCASilv=rep(NA,it)
	threeDiffDiameterPCADip=rep(NA,it)
	threeDiffDensitySilv=rep(NA,it)
	threeDiffDensityDip=rep(NA,it)
	threeDiffDensityHop=rep(NA,it)
	threeDiffDensityClassicSilv=rep(NA,it)
	threeDiffDensityClassicDip=rep(NA,it)
	threeDiffDensityPCASilv=rep(NA,it)
	threeDiffDensityPCADip=rep(NA,it)
	twoSep10DSilv=rep(NA,it)
	twoSep10DDip=rep(NA,it)
	twoSep10DHop=rep(NA,it)
	twoSep10DClassicSilv=rep(NA,it)
	twoSep10DClassicDip=rep(NA,it)
	twoSep10DPCASilv=rep(NA,it)
	twoSep10DPCADip=rep(NA,it)
	fourSep10DSilv=rep(NA,it)
	fourSep10DDip=rep(NA,it)
	fourSep10DHop=rep(NA,it)
	fourSep10DClassicSilv=rep(NA,it)
	fourSep10DClassicDip=rep(NA,it)
	fourSep10DPCASilv=rep(NA,it)
	fourSep10DPCADip=rep(NA,it)
	
	threeRowSilv=rep(NA,it)
	threeRowDip=rep(NA,it)
	threeRowHop=rep(NA,it)
	threeRowClassicSilv=rep(NA,it)
	threeRowClassicDip=rep(NA,it)
	threeRowPCASilv=rep(NA,it)
	threeRowPCADip=rep(NA,it)
	
	fiveRowSilv=rep(NA,it)
	fiveRowDip=rep(NA,it)
	fiveRowHop=rep(NA,it)
	fiveRowClassicSilv=rep(NA,it)
	fiveRowClassicDip=rep(NA,it)
	fiveRowPCASilv=rep(NA,it)
	fiveRowPCADip=rep(NA,it)

	sevenRowSilv=rep(NA,it)
	sevenRowDip=rep(NA,it)
	sevenRowHop=rep(NA,it)
	sevenRowClassicSilv=rep(NA,it)
	sevenRowClassicDip=rep(NA,it)
	sevenRowPCASilv=rep(NA,it)
	sevenRowPCADip=rep(NA,it)

	for(i in 1:it){
		loopTimer <- proc.time()		#starts loop stopwatch

		#Create
		threeSepClust<-threeCLOSERSeparatedClusters()
		threeSepClust3D<-threeCLOSERSeparatedClusters3D()
		threeNoisyClust<-threeCLOSERNoisyClusters()
		threeClustDiffDiameter<-threeCLOSERClustersVaryingDiameter()
		threeClustDiffDensity<-threeCLOSERClustersVaryingDensity()
		twoSepClust10D<-twoCLOSERSeparatedClusters10D()	
		fourSepClust10D<-fourCLOSERSeparatedClusters10D()
		threeRow<-threeRowClusters()
		fiveRow<-fiveRowClusters()
		sevenRow<-sevenRowClusters()


		#dist
		threeSepSilv[i]<-silverman.test(dist(threeSepClust),1,adjust=TRUE)@p_value
		threeSep3DSilv[i]<-silverman.test(dist(threeSepClust3D),1,adjust=TRUE)@p_value
		threeNoisySilv[i]<-silverman.test(dist(threeNoisyClust),1,adjust=TRUE)@p_value
		threeDiffDiameterSilv[i]<-silverman.test(dist(threeClustDiffDiameter),1,adjust=TRUE)@p_value
		threeDiffDensitySilv[i]<-silverman.test(dist(threeClustDiffDensity),1,adjust=TRUE)@p_value
		twoSep10DSilv[i]<-silverman.test(dist(twoSepClust10D),1,adjust=TRUE)@p_value
		fourSep10DSilv[i]<-silverman.test(dist(fourSepClust10D),1,adjust=TRUE)@p_vaue
		threeRowSilv[i]<-silverman.test(dist(threw),1,adjust=TRUE)@p_value
		fiveRowSilv[i]<-silverman.test(dist(fiveRow),1,adjust=TRUE)@p_value
		sevenRowSilv[i]<-silverman.test(dist(sevenRow),1,adjust=TRUE)@p_value

		
		threeSepDip[i]<-dip.test(dist(threeSepClust))$p.value
		threeSep3DDip[i]<-dip.test(dist(threeSepClust3D))$p.value
		threeNoisyDip[i]<-dip.test(dist(threeNoisyClust))$p.value
		threeDiffDiameterDip[i]<-dip.test(dist(threeClustDiffDiameter))$p.value
		threeDiffDensityDip[i]<-dip.test(dist(threeClustDiffDensity))$p.value
		twoSep10DDip[i]<-dip.test(dist(twoSepClust10D))$p.value
		fourSep10DDip[i]<-dip.test(dist(fourSepClust10D))$p.value
		threeRowDip[i]<-dip.test(dist(threeRow))$p.value
		fiveRowDip[i]<-dip.test(dist(fiveRow))$p.value
		sevenRowDip[i]<-dip.test(dist(sevenRow))$p.value
		
	#Hop
		threeSepHop[i]<-hopkins(as.data.frame(threeSepClust), n = hopN)$H
		threeSep3DHop[i]<-hopkins(as.data.frame(threeSepClust3D), n = hopN)$H
		threeNoisyHop[i]<-hopkins(as.data.frame(threeNoisyClust), n = hopN)$H
		threeDiffDiameterHop[i]<-hopkins(as.data.frame(threeClustDiffDiameter), n = hopN)$H
		threeDiffDensityHop[i]<-hopkins(as.data.frame(threeClustDiffDensity), n = hopN)$H
		twoSep10DHop[i]<-hopkins(as.data.frame(twoSepClust10D), n = hopN)$H
		fourSep10DHop[i]<-hopkins(as.data.frame(fourSepClust10D), n = hopN)$H
		threeRowHop[i]<-hopkins(as.data.frame(threeRow), n = hopN)$H
		fiveRowHop[i]<-hopkins(as.data.frame(fiveRow), n = hopN)$H
		sevenRowHop[i]<-hopkins(as.data.frame(sevenRow), n = hopN)$H
		
	#Classic
		threeSepClassicDip[i]<-dip.test(threeSepClust)$p.value
		threeSep3DClassicDip[i]<-dip.test(threeSepClust3D)$p.value
		threeNoisyClassicDip[i]<-dip.test(threeNoisyClust)$p.value
		threeDiffDiameterClassicDip[i]<-dip.test(threeClustDiffDiameter)$p.value
		threeDiffDensityClassicDip[i]<-dip.test(threeClustDiffDensity)$p.value
		twoSep10DClassicDip[i]<-dip.test(twoSepClust10D)$p.value
		fourSep10DClassicDip[i]<-dip.test(fourSepClust10D)$p.value
		threeRowClassicDip[i]<-dip.test(threeRow)$p.value
		fiveRowClassicDip[i]<-dip.test(fiveRow)$p.value
		sevenRowClassicDip[i]<-dip.test(sevenRow)$p.value

		threeSepClassicSilv[i]<-silverman.test(threeSepClust,1,adjust=TRUE)@p_value
		threeSep3DClassicSilv[i]<-silverman.test(threeSepClust3D,1,adjust=TRUE)@p_value
		threeNoisyClassicSilv[i]<-silverman.test(threeNoisyClust,1,adjust=TRUE)@p_value
		threeDiffDiameterClassicSilv[i]<-silverman.test(threeClustDiffDiameter,1,adjust=TRUE)@p_value
		threeDiffDensityClassicSilv[i]<-silverman.test(threeClustDiffDensity,1,adjust=TRUE)@p_value
		twoSep10DClassicSilv[i]<-silverman.test(twoSepClust10D,1,adjust=TRUE)@p_value
		fourSep10DClassicSilv[i]<-silverman.test(fourSepClust10D,1,adjust=TRUE)@p_value
		threeRowClassicSilv[i]<-silverman.test(threeRow,1,adjust=TRUE)@p_value
		fiveRowClassicSilv[i]<-silverman.test(fiveRow,1,adjust=TRUE)@p_value
		sevenRowClassicSilv[i]<-silverman.test(sevenRow,1,adjust=TRUE)@p_value

					
	#PCA	
		threeSepPCASilv[i]<-silverman.test(prcomp(threeSepClust)$x[,1],1,adjust=TRUE)@p_value
		threeSep3DPCASilv[i]<-silverman.test(prcomp(threeSepClust3D)$x[,1],1,adjust=TRUE)@p_value
		threeNoisyPCASilv[i]<-silverman.test(prcomp(threeNoisyClust)$x[,1],1,adjust=TRUE)@p_value
		threeDiffDiameterPCASilv[i]<-silverman.test(prcomp(threeClustDiffDiameter)$x[,1],1,adjust=TRUE)@p_value
		threeDiffDensityPCASilv[i]<-silverman.test(prcomp(threeClustDiffDensity)$x[,1],1,adjust=TRUE)@p_value
		twoSep10DPCASilv[i]<-silverman.test(prcomp(twoSepClust10D)$x[,1],1,adjust=TRUE)@p_value
		fourSep10DPCASilv[i]<-silverman.test(prcomp(fourSepClust10D)$x[,1],1,adjust=TRUE)@p_value
		threeRowPCASilv[i]<-silverman.test(prcomp(threeRow)$x[,1],1,adjust=TRUE)@p_value
		fiveRowPCASilv[i]<-silverman.test(prcomp(fiveRow)$x[,1],1,adjust=TRUE)@p_value
		sevenRowPCASilv[i]<-silverman.test(prcomp(sevenRow)$x[,1],1,adjust=TRUE)@p_value		

		threeSepPCADip[i]<-dip.test(prcomp(threeSepClust)$x[,1])$p.value
		threeSep3DPCADip[i]<-dip.test(prcomp(threeSepClust3D)$x[,1])$p.value
		threeNoisyPCADip[i]<-dip.test(prcomp(threeNoisyClust)$x[,1])$p.value
		threeDiffDiameterPCADip[i]<-dip.test(prcomp(threeClustDiffDiameter)$x[,1])$p.value
		threeDiffDensityPCADip[i]<-dip.test(prcomp(threeClustDiffDensity)$x[,1])$p.value
		twoSep10DPCADip[i]<-dip.test(prcomp(twoSepClust10D)$x[,1])$p.value
		fourSep10DPCADip[i]<-dip.test(prcomp(fourSepClust10D)$x[,1])$p.value
		threeRowPCADip[i]<-dip.test(prcomp(threeRow)$x[,1])$p.value
		fiveRowPCADip[i]<-dip.test(prcomp(fiveRow)$x[,1])$p.value
		sevenRowPCADip[i]<-dip.test(prcomp(sevenRow)$x[,1])$p.value
	

	#feedback
		loopClock<-as.numeric(procc.time()-loopTimer)[3]		#stop loopstopwatch, and extract elapsed time
		clock=clock+loopClock						#add to total time passed.
		clockRemain=((3*(clock/i)+(loopClock))/4)*(it-i)		#predict future based on average of current simulation speed vs history

		cat(" Finished Generation: ", i, "/", it)
		printTime(prefix=" in ", loopClock)
		printTime(prefix="Total Time Elapsed: ", clock)  
		printTime(prefix="Estimated Time Remaining: ",clockRemain,suffix="\n")	
		flush.console()
		Sys.sleep(0.001)
	}

	result<-matrix(,10,8)

	colnames(result)<-c("Dip","Silv.","Hopkins Test","Hop<0.25","Classic Dip","Classic Silv.","PCA Dip", "PCA Silv.")
	rownames(result)<-c("threeSepClust","threeSepClust3D","threeNoisyClust",
			"threeClustDiffDiameter","threeClustDiffDensity",
			"twoSepClust10D","fourSepClust10D",
			"threeRow", "fiveRow", "sevenRow1")

	result[1,1]<-sum(threeSepDip<0.05)/it
	result[1,2]<-sum(threeSepSilv<0.05)/it
	result[1,3]<-round(mean(threeSepHop),4)
	result[1,4]<-round(sum(threeSepHop<0.25)/it,4)
	result[1,5]<-sum(threeSepClassicDip<0.05)/it
	result[1,6]<-sum(threeSepClassicSilv<0.05)/it
	result[1,7]<-sum(threeSepPCADip<0.05)/it
	result[1,8]<-sum(threeSepPCASilv<0.05)/it

	result[2,1]<-sum(threeSep3DDip<0.05)/it
	result[2,2]<-sum(threeSep3DSilv<0.05)/it
	result[2,3]<-round(mean(threeSep3DHop),4)
	result[2,4]<-round(sum(threeSep3DHop<0.25)/it,4)
	result[2,5]<-sum(threeSep3DClassicDip<0.05)/it
	result[2,6]<-sum(threeSep3DClassicSilv<0.05)/it
	result[2,7]<-sum(threeSep3DPCADip<0.05)/it
	result[2,8]<-sum(threeSep3DPCASilv<0.05)/it

	result[3,1]<-sum(threeNoisyDip<0.05)/it
	result[3,2]<-sum(threeNoisySilv<0.05)/it
	result[3,3]<-round(mean(threeNoisyHop),4)
	result[3,4]<-round(sum(threeNoisyHop<0.25)/it,4)
	result[3,5]<-sum(threeNoisyClassicDip<0.05)/it
	result[3,6]<-sum(threeNoisyClassicSilv<0.05)/it
	result[3,7]<-sum(threeNoisyPCADip<0.05)/it
	result[3,8]<-sum(threeNoisyPCASilv<0.05)/it

	result[4,1]<-sum(threeDiffDiameterDip<0.05)/it
	result[4,2]<-sum(threeDiffDiameterSilv<0.05)/it
	result[4,3]<-round(mean(threeDiffDiameterHop),4)
	result[4,4]<-round(sum(threeDiffDiameterHop<0.25)/it,4)
	result[4,5]<-sum(threeDiffDiameterClassicDip<0.05)/it
	result[4,6]<-sum(threeDiffDiameterClassicSilv<0.05)/it
	result[4,7]<-sum(threeDiffDiameterPCADip<0.05)/it
	result[4,8]<-sum(threeDiffDiameterPCASilv<0.05)/it

	result[5,1]<-sum(threeDiffDensityDip<0.05)/it
	result[5,2]<-sum(threeDiffDensitySilv<0.05)/it
	result[5,3]<-round(mean(threeDiffDensityHop),4)
	result[5,4]<-round(sum(threeDiffDensityHop<0.25)/it,4)
	result[5,5]<-sum(threeDiffDensityClassicDip<0.05)/it
	result[5,6]<-sum(threeDiffDensityClassicSilv<0.05)/it
	result[5,7]<-sum(threeDiffDensityPCADip<0.05)/it
	result[5,8]<-sum(threeDiffDensityPCASilv<0.05)/it

	result[6,1]<-sum(twoSep10DDip<0.05)/it
	result[6,2]<-sum(twoSep10DSilv<0.05)/it
	result[6,3]<-round(mean(twoSep10DHop),4)
	result[6,4]<-round(sum(twoSep10DHop<0.25)/it,4)
	result[6,5]<-sum(twoSep10DClassicDip<0.05)/it
	result[6,6]<-sum(twoSep10DClassicSilv<0.05)/it
	result[6,7]<-sum(twoSep10DPCADip<0.05)/it
	result[6,8]<-sum(twoSep10DPCASilv<0.05)/it

	result[7,1]<-sum(fourSep10DDip<0.05)/it
	result[7,2]<-sum(fourSep10DSilv<0.05)/it
	result[7,3]<-round(mean(fourSep10DHop),4)
	result[7,4]<-round(sum(fourSep10DHop<0.25)/it,4)
	result[7,5]<-sum(fourSep10DClassicDip<0.05)/it
	result[7,6]<-sum(fourSep10DClassicSilv<0.05)/it
	result[7,7]<-sum(fourSep10DPCADip<0.05)/it
	result[7,8]<-sum(fourSep10DPCASilv<0.05)/it

	result[8,1]<-sum(threeRowDip<0.05)/it
	result[8,2]<-sum(threeRowSilv<0.05)/it
	result[8,3]<-round(mean(threeRowHop),4)
	result[8,4]<-round(sum(threeRowHop<0.25)/it,4)
	result[8,5]<-sum(threeRowClassicDip<0.05)/it
	result[8,6]<-sum(threeRowClassicSilv<0.05)/it
	result[8,7]<-sum(threeRowPCADip<0.05)/it
	result[8,8]<-sum(threeRowPCASilv<0.05)/it

	result[9,1]<-sum(fiveRowDip<0.05)/it
	result[9,2]<-sum(fiveRowSilv<0.05)/it
	result[9,3]<-round(mean(fiveRowHop),4)
	result[9,4]<-round(sum(fiveRowHop<0.25)/it,4)
	result[9,5]<-sum(fiveRowClassicDip<0.05)/it
	result[9,6]<-sum(fiveRowClassicSilv<0.05)/it
	result[9,7]<-sum(fiveRowPCADip<0.05)/it
	result[9,8]<-sum(fiveRowPCASilv<0.05)/it

	result[10,1]<-sum(sevenRowDip<0.05)/it
	result[10,2]<-sum(sevenRowSilv<0.05)/it
	result[10,3]<-round(mean(sevenRowHop),4)
	result[10,4]<-round(sum(sevenRowHop<0.25)/it,4)
	result[10,5]<-sum(sevenRowClassicDip<0.05)/it
	result[10,6]<-sum(sevenRowClassicSilv<0.05)/it
	result[10,7]<-sum(sevenRowPCADip<0.05)/it
	result[10,8]<-sum(sevenRowPCASilv<0.05)/it

	return(result)
}

main<-function(iterations = 1000, setSeed = TRUE, seed = 42){
	if(setSeed == TRUE){
		set.seed(seed)
	}
	
	#quickRun(iterations)
	
	#generatedData(iterations)
	#cat("\n\n\n")					#line separator
	#realData(100)
	oneRun(1000)
}

main()

