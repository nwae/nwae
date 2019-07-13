#!/bin/bash

#
# Load test to bombard our chat server
#

# We will random select parts of the text to send
HUAWEI="
美国政府坚持说，美国反对在关键信息系统中使用华为技术，完全是出于对安全的担心——比如华为可能在其产品中安装“后门”，以及中国政府与中国高技术公司的密切关系等。

当然，两国之间的商业竞争可能也是一个重要原因。毕竟，信息技术对全世界的经济前景至关重要。

但是，是否还有其他因素？这会不会是一场远远超出传统贸易战的巨大博弈，而我们目前看到的只是其第一场战役？

美国科技禁令：中国科技企业滑铁卢还是世界秩序新起点
华为"封杀令"：芯片公司ARM要求员工停止与中国科技巨头合作
华为被断供后 美国供应商可能承受的痛
多年以来，中国的崛起，以及随之而来的世界经济重心东移、美国国力相对衰退，早已成为国际评论家的谈资。但这些趋势本来就难免造成摩擦。现在，美国已经开始反击。

美国反击
美国政府发言人开始将“全球竞争的新时代”挂在嘴边。最初的焦点集中在军事上——美军不再将反恐和局部战争作为重点，而是开始为大国之间的军事冲突作准备，并将俄罗斯和中国视为竞争对手。

但是，在美国与中国的争斗之中，经济是一个根本因素。特朗普（川普）政府似乎已下决心使用美国的经济力量，不仅要限制华为这样的中国公司，还要迫使北京开放国内市场，并改变其长期以来饱受在华西方企业诟病的监管行为。

在北京看来，这是美国在试图遏制中国的崛起。这种看法可能是正确的。

但这场争斗涉及的决不只是经济行为和商业市场。这是一场关乎两国国力之根基的搏斗，将产生巨大的战略影响。换句话说，西方正在渐渐重新认识一条基本的法则——经济实力是国家力量的基础，也是军事实力的前提。而北京对此法则早已了然于心。

华为遭封杀：中国呼吁英国考虑自身国家利益
华为遭断供：失去谷歌三宝的手机用户日子怎么过
"

# Generate random Chat ID using machine name and date time
CHATID="`uname`: `date`"
# Convert to base 64
CHATID=`echo "$CHATID" | base64 -i -`

BIBOT_SERVER="http://13.231.33.215:5000/intent"
BIBOT_SERVER="http://localhost:5000/intent"

echo "Load test to server $BIBOT_SERVER start..."

# Get string length
SLEN=${#HUAWEI}
# Request string length
# The reason we pick 10 is because from our thousands of training data,
# the 25%, 50%, 75% quartile for the text length is 7, 10, 14 respectively
REQ_STR_LEN=10
# Random length, SLEN-10
RLEN=$((SLEN - REQ_STR_LEN - 1))

for i in {1..1000}
    do
        # Generate random position
        POS=$RANDOM
        # Make sure this number is within our random length
        POS=$(($POS % RLEN))
        RANDOM_STRING=${HUAWEI:POS:10}
        # Replace newlines
        RANDOM_STRING=`echo $RANDOM_STRING | sed s/"[\n\r]"/' '/g`
        echo "$i-th request to server.. $RANDOM_STRING"

        REQDATA="{
           \"accid\": \"3\",
           \"botid\": \"4\",
           \"chatid\": \"$CHATID\",
           \"txt\": \"$RANDOM_STRING\"
         }"

        echo "$REQDATA"

        curl -i -X POST \
            -H 'Content-Type: application/json' \
            -d "$REQDATA" \
            $BIBOT_SERVER

         sleep 0.1
         echo ""
    done
