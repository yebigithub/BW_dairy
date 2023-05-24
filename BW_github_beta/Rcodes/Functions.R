# Functions to be used in prediction ----
# Useful useful
## For prediction ----
preprosess_func <- function(method){
  ####
  #function to be used to preprocess data.df
  #Function to be used in the beginning of cv predictions.
  ## change CowID into 1:12
  ## change day from numeric into D+numeric
  ## Include FullID as one new column.
  ## Remove outlier of real BW.
  
  cat("## Done with preprocessing for preprosess_func \n")
  load(paste0("/Users/yebi/Library/CloudStorage/OneDrive-VirginiaTech/Research/Codes/research/BCS/BodyWeight/outputs_R/outputs_", method, "/bw_img.L.RData"))
  data.df = do.call(rbind, bw_img.L)
  
  for (i in 1:12){
    cowids = as.vector(unique(data.df$ID))
    data.df$ID[data.df$ID == cowids[i]] = gsub(cowids[i], i, data.df$ID[data.df$ID == cowids[i]])
  }
  
  data.df$day = as.numeric(gsub("D", "", data.df$DAY))
  data.df$FullID = paste0(data.df$DAY,"_", data.df$FullID)
  rownames(data.df) = data.df$FullID
  
  data.df$day_time = paste0(data.df$DAY, data.df$Time)
  data.df = data.df %>% filter(day_time != "D5AM") %>% droplevels()
  
  for (i in 1:54){
    day_times = as.vector(unique(data.df$day_time))
    data.df$day_time_num[data.df$day_time == day_times[i]] = gsub(day_times[i], i, data.df$day_time[data.df$day_time == day_times[i]])
  }
  
  data.df$day_time_num = as.numeric(data.df$day_time_num)
  data.df = data.df[!data.df$BW<200, ] #remove outlier in BW
  data.df = na.omit(data.df) # remove NA in the whole dataframe

  data.df$ID = factor(data.df$ID, levels = 1:12)
  return(data.df)
  
}

load_func <- function(img_method, pred_method){
  ### function used to load prediction results for different img_method and pred_method
  
  filename = paste0("/Users/yebi/Library/CloudStorage/OneDrive-VirginiaTech/Research/Codes/research/BCS/BodyWeight/outputs_R/outputs_", img_method, "/", pred_method, "/CV2_predr.RData")
  load(filename)
  return(pred.df)
}


cv2_point_plot_func <- function(method, ID_tst, name){
  
  ### Prediction CV2
  ### main function to draw plots for cv2 bottom/top 5.
  
  cat("## Drawing cv3 cv plots \n")
  
  data.df = preprosess_func(method)
  IDs = as.character(1:12)
  ID_trn = IDs[!IDs%in%ID_tst]
  data.df$ID = factor(data.df$ID, levels = c(ID_trn, ID_tst))
  data.df$CV[data.df$ID %in% ID_tst] = "Test"
  data.df$CV[!(data.df$ID %in% ID_tst)] = "Train"
  data.df$CV = as.factor(data.df$CV)
  # plot(BW ~ ID, data = data.df)
  ggplot(data.df, aes(x=ID, y=BW, color = CV))+
    geom_jitter(size = 1, width = 0.25)+
    # ggtitle(name)+
    scale_color_manual(values=c('#b2182b', '#2166ac'))+
    # scale_color_brewer(palette="RdBu")+
    ylab("Body weight (kg)")+
    xlab("Cow ID")+
    theme_classic()+
    theme(axis.title.x = element_text(size = 12, margin = margin(t = 3)),
          axis.title.y = element_text(size = 12, margin = margin(r = 3)),
          axis.text.x = element_text(size = 12, hjust=0.5, vjust=0.2), 
          axis.text.y = element_text(size = 12),
          legend.position = "none")+
    theme(plot.margin = margin(1,1,1,1, "cm"))
  
}


top_bot_5_func <- function(r3, decide){
  
  ## function to find testing and train sets for top and bottom 5 cv2 prediction results
  
  cat("## Done with top bottom 4 ID finding \n")
  if(decide == "R2"){
    top.df = r3 %>% arrange(desc(R2))
    bot.df = r3 %>% arrange(R2) 
  }
  else{
    top.df = r3 %>% arrange(MAPE)
    bot.df = r3 %>% arrange(desc(MAPE))
  }
  
  
  comb.top = data.frame(top.df[1:4,4:6], order = "TOP")
  comb.bot = data.frame(bot.df[1:4,4:6], order = "BOT") 
  return(rbind.data.frame(comb.top, comb.bot))
}



plot_draw_function <- function(method = "ST", decide, orr = "TOP"){
  
  #function used to draw plots for top and bottom 5
  
  
  if(method == "ST"){r3 = r3.sig.thr}
  if(method == "AT"){r3 = r3.adp.thr}
  if(method =="MRCNN"){r3 = r3.mrcnn}
  ###########################################################
  ## Top5
  ###########################################################
  
  plot_top.L = list()
  for(i in 1:4){
    top.df = as.vector(top_bot_5_func(r3, decide = decide)[i, ])
    
    ID_tst = as.character(top.df[1:3])
    name = as.character(top.df[4])
    
    plot_top.L[[i]] = cv2_point_plot_func(
      method = method,
      ID_tst = ID_tst,
      name = paste0(name, "-", i)
    )
  }
  if(orr == "TOP"){
  return(ggpubr::ggarrange(plot_top.L[[1]],
                          plot_top.L[[2]],
                          plot_top.L[[3]],
                          plot_top.L[[4]],
                          ncol = 2, 
                          nrow = 2))
  }
  # dev.print(pdf, file=paste0("./outputs/plot_top5", "_", method,".pdf"), 
  #           height = 8, width = 12)
  
  ###########################################################
  ## bottom 5
  ###########################################################  
  plot_bot.L = list()
  
  for(i in 5:8){
    bot.df = as.vector(top_bot_5_func(r3, decide = decide)[i, ])
    
    ID_tst = as.character(bot.df[1:3])
    name = as.character(bot.df[4])
    
    plot_bot.L[[i-4]] = cv2_point_plot_func(
      method = method,
      ID_tst = ID_tst,
      name = paste0(name, "-", i-5)
    )
  }
  if(orr != "TOP"){
  return(ggpubr::ggarrange(plot_bot.L[[1]],
                          plot_bot.L[[2]],
                          plot_bot.L[[3]],
                          plot_bot.L[[4]],
                          ncol = 2, 
                          nrow = 2))
  }
  # dev.print(pdf, file=paste0("./outputs/plot_bot5", "_", method,".pdf"), 
  #           height = 8, width = 12)
}
