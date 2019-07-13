# Clear memory
rm(list=ls())

WORKING.DIR <- "~/dev/ie/ie.chats"
# Change directory
setwd(WORKING.DIR)

# Avoid scientific notation such as 1e+05, so that when in Excel it will be exported correctly to DB
options(scipen=999)

require("RPostgreSQL")
options(Encoding="UTF-8")

psqlGetConnection <- function(dbhost    = "rstudio.nettsys.tech",
                              dbport    = 5432,
                              dbname    = "calllogdb",
                              dbuser    = intToUtf8(c(9592,8536,10032,9416,4048,10208,8536,9681)/88),
                              dbuserpwd = intToUtf8(c(7392,9768,9416,10648,9768,4928,4928,5016)/88)
                              )
{
  # loads the PostgreSQL driver
  drv <- dbDriver("PostgreSQL")
  # creates a connection to the postgres database
  # note that "con" will be used later in each connection to the database
  con <- dbConnect(drv, dbname = dbname,
                   host = dbhost, port = dbport,
                   user = dbuser, password = dbuserpwd)
  return(con)
}

getdata.chats.analyticdb <- function(brand, currency, datefrom, dateto)
{
  con <- psqlGetConnection()

  sql <- paste("SELECT *",
               " FROM \"Calllog\"",
               " WHERE chatdatetime>='", datefrom, "' AND chatdatetime<='", dateto, "'",
               " AND brand='", brand, "'",
               " AND currency='", currency, "'",
               sep="")

  chats <- dbGetQuery(conn = con, sql)

  dbDisconnect(conn = con)

  if(dim(chats)[1] == 0) stop("No Data from DB!")

  # Convert to POSIX Datetime
  chats$DateTime <- as.POSIXlt(chats$chatdatetime, format="%Y-%m-%d %H:%M:%S")

  # Convert Line to numeric
  chats$line <- as.numeric(chats$line)

  # Extract relevant columns and order properly
  columnskeep <- c("LivePersonSessionID", "brand", "currency", "DateTime", "line", "category", "speaker", "content")
  chatsret <- chats[,columnskeep]
  chatsret <- chatsret[order(chatsret$LivePersonSessionID, chatsret$line, decreasing=FALSE),]

  # Get pause during chat lines
  colsreq <- c("LivePersonSessionID", "DateTime", "line")
  chatprev <- rbind(chatsret[1,colsreq], chatsret[1:(dim(chatsret)[1]-1),colsreq])
  colnames(chatprev) <- c("PrevID", "PrevDateTime", "PrevLine")
  chatprev$PrevID[1] <- "NA"

  chatsret <- cbind(chatsret, chatprev)
  # Get seconds between chat lines
  chatsret$Pause <- as.numeric(chatsret$DateTime - chatsret$PrevDateTime)
  # NA for those of different chats
  chatsret$Pause[chatsret$LivePersonSessionID!=chatsret$PrevID] <- NA

  colsfinal <- c("LivePersonSessionID", "brand", "currency", "DateTime", "Pause", "line", "category", "speaker", "content")
  chatsretfinal <- chatsret[,colsfinal]

  return(chatsretfinal)
}

#
# From Unified Chat Table
#
getdata.chats.analyticdb.utchat <- function(brand, currency, datefrom, dateto)
{
  con <- psqlGetConnection()
  
  sql <- paste("SELECT *",
               " FROM \"calllogut\"",
               " WHERE \"Chatdate\">='", datefrom, "' AND \"Chatdate\"<='", dateto, "'",
               " AND \"Brand\"='", brand, "'",
               " AND \"Currency\"='", currency, "'",
               sep="")
  
  chats <- dbGetQuery(conn = con, sql)
  
  dbDisconnect(conn = con)
  
  if(dim(chats)[1] == 0) stop("No Data from DB!")
  
  # Convert to POSIX Datetime
  chats$DateTime <- as.POSIXlt(chats$Chatdate, format="%Y-%m-%d %H:%M:%S")
  
  # Convert Line to numeric
  chats$Line <- as.numeric(chats$Line)
  
  # Extract relevant columns and order properly
  columnskeep <- c("CalllogID", "Brand", "Currency", "DateTime", "Line", "Category", "Speaker", "Content")
  chatsret <- chats[,columnskeep]
  chatsret <- chatsret[order(chatsret$CalllogID, chatsret$Line, decreasing=FALSE),]
  
  # Get pause during chat lines
  colsreq <- c("CalllogID", "DateTime", "Line")
  chatprev <- rbind(chatsret[1,colsreq], chatsret[1:(dim(chatsret)[1]-1),colsreq])
  colnames(chatprev) <- c("PrevID", "PrevDateTime", "PrevLine")
  chatprev$PrevID[1] <- "NA"
  
  chatsret <- cbind(chatsret, chatprev)
  # Get seconds between chat lines
  chatsret$Pause <- as.numeric(chatsret$DateTime - chatsret$PrevDateTime)
  # NA for those of different chats
  chatsret$Pause[chatsret$CalllogID!=chatsret$PrevID] <- NA
  
  colsfinal <- c("CalllogID", "Brand", "Currency", "DateTime", "Pause", "Line", "Category", "Speaker", "Content")
  chatsretfinal <- chatsret[,colsfinal]
  
  return(chatsretfinal)
}

#
# Get chat data in standard format
#
getdata.chats <- function(source="analyticdb", path="", brand, currency, datefrom, dateto,)
{
  print(paste("Fetching data from [", source, "] for [", brand, ":", currency, " ", datefrom, " to ", dateto, "]", sep=""))

  if(source == "analyticdb") {
    calllog <- getdata.chats.analyticdb(brand=brand, currency=currency, datefrom=datefrom, dateto=dateto)
  }
  else if (source == "analyticdb.unifiedtable") {
    calllog <- getdata.chats.analyticdb.utchat(brand=brand, currency=currency, datefrom=datefrom, dateto=dateto)
  }
  else if(source == "file") {
    filepath <- paste("../app.data/chatdata/", brand, ".", currency, ".", datefrom, ".to.", dateto, ".csv", sep="")
    calllog <- read.csv(file=filepath, header=TRUE)
    calllog$DateTime <- as.POSIXlt(calllog$DateTime, format="%Y-%m-%d %H:%M:%S")
  }
  else {
    stop("Source [", source, "] not supported!")
  }

  return(calllog)
}

get.data.test <- function()
{
  source="analyticdb.unifiedtable"
  brand="TLC"
  currency="CNY"
  datefrom="2018-09-01"
  dateto="2018-09-10"
  calllog <- getdata.chats(source=source, brand=brand, currency=currency, datefrom=datefrom, dateto=dateto)
  write.csv(calllog, paste("../app.data/chatdata/", brand, ".", currency, ".", datefrom, ".to.", dateto, ".csv", sep=""), row.names = FALSE)

  source = "file"
  calllog.file <- getdata.chats(source=source, brand=brand, currency=currency, datefrom=datefrom, dateto=dateto)
}

#get.data.test()

