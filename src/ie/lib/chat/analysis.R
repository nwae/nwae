#
# Various Analysis on our chats
#

rm(list=ls())

WD = "/Users/mark.tan/dev/ie/py.ie.lib/chat"
CHAT.CLUSTERING.DIR = "/Users/mark.tan/dev/ie/app.data/chat.clustering"
FNAME = 'Betway.CNY.2018-01-01.to.2018-01-31.membervisitor.line.01.csv'

df.raw = read.csv(file=paste(CHAT.CLUSTERING.DIR,"/",FNAME, sep=""), sep = ",")

# Remove those with score confidence <=2, with no Intent
df <- df.raw[df.raw$Intent!="-" & df.raw$IntentScoreConfidence>=2,]
df$Count <- 1
df.agg = aggregate(x=df$Count, by=list("Intent"=df$Intent), FUN="sum")

df.agg = df.agg[order(df.agg$x, decreasing = TRUE),]
df.agg$Proportion = round(100 * df.agg$x / sum(df.agg$x), 2)
