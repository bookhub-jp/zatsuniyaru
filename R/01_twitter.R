library(tidyverse)
library(feather)
library(rtweet)
library(tangela)
library(quanteda)
library(ggpubr)
library(recommenderlab)
library(arules)
library(arulesViz)

tangela::start_tangela(dotenv = file.path("./dotenv.env"))

#### データの準備 ####
rt <- search_tweets(
    "レッドフレーム改",
    n = 2000,
    include_rts = FALSE,
    retryonratelimit = TRUE
) %>%
    select(created_at, text) %>%
    write_feather("tweets.feather")

rt <- read_feather("tweets.feather")

rt$text <- rt$text %>%
    str_replace_all(regex("[’]"), "\'") %>%
    str_replace_all(regex("[”]"), "\"") %>%
    str_replace_all(regex("[˗֊‐‑‒–⁃⁻₋−]+"), "-") %>%
    str_replace_all(regex("[﹣－ｰ—―─━ー]+"), "ー") %>%
    str_remove_all(regex("[~∼∾〜〰～]")) %>%
    str_remove_all("[:punct:]") %>%
    str_remove_all("[:blank:]") %>%
    str_remove_all("[:cntrl:]") %>%
    Nippon::zen2han()

rt$tokens <- tangela::tokenize(
    docs = as.list(rt$text),
    host = "127.0.0.8",
    port = 3033
) %>%
    map(~ map(., ~ .x$surface_form)) %>%
    map(~ paste(., collapse = " ")) %>%
    purrr::flatten() %>%
    unlist()

corp <- rt %>%
    distinct(tokens, .keep_all = TRUE) %>%
    corpus(text_field = "tokens")

#### KWIC ####
toks <- tokens(corp, what = "fastestword", remove_punct = TRUE, remove_twitter = TRUE)

kwic(toks, "レッドフレーム") %>%
    as_tibble() %>%
    View()

textstat_collocations(toks, size = 2) %>%
    as_tibble() %>%
    View()

#### グラフの描画 ####
multiword <- c("レッドフレーム 改", "斗和 キセキ")

tokens(corp, what = "fastestword", remove_punct = TRUE, remove_twitter = TRUE) %>%
    tokens_compound(pattern = phrase(multiword)) %>%
    tokens_skipgrams(n = 3, skip = 0) %>%
    dfm() %>%
    textplot_wordcloud(
        min_count = 10,
        random_order = FALSE,
        color = viridisLite::cividis(8)
    )

tokens(corp, what = "fastestword", remove_punct = TRUE, remove_twitter = TRUE) %>%
    tokens_compound(pattern = phrase(multiword)) %>%
    tokens_skipgrams(n = 3, skip = 0) %>%
    dfm() %>%
    fcm() %>%
    fcm_select(., names(topfeatures(., 150))) %>%
    textplot_network(min_freq = 0.95, edge_size = 5)

#### トピックモデル ####
toks <- tokens_compound(toks, pattern = phrase(multiword))

mx <- dfm(toks) %>%
    dfm_trim(min_termfreq = 10L) %>%
    dfm_remove(tangela::StopWordsJp$word) %>%
    dfm_remove(tangela::ExtendedLettersJp$letter) %>%
    dfm_remove(tangela::OneLettersJp$letter) %>%
    dfm_remove("[0-9]", valuetype = "regex")

topicmdl <- mx %>%
    convert(to = "topicmodels") %>%
    topicmodels::LDA(k = 2, method = "Gibbs")

topicdoc <- tidytext::tidy(topicmdl, matrix = "gamma") %>%
    group_by(document) %>%
    top_n(n = 1, wt = gamma) %>%
    arrange(topic) %>% 
    ungroup() %>%
    as_tibble()

ggpubr::ggviolin(
    topicdoc,
    x = "topic",
    y = "gamma",
    color = "topic",
    add = "jitter",
    shape = "topic"
)

topicterm <- tidytext::tidy(topicmdl, matrix = "beta") %>%
    group_by(term) %>%
    top_n(n = 1, wt = beta) %>%
    arrange(topic) %>% 
    ungroup() %>%
    group_by(topic) %>%
    top_n(n = 25, wt = beta) %>%
    arrange(desc(beta)) %>%
    as_tibble()

topicterm$topic <- as.factor(topicterm$topic)

ggpubr::ggbarplot(
    topicterm,
    x = "term",
    y = "beta",
    color = "white",
    fill = "topic",
    sort.val = "asc",
    sort.by.groups = TRUE,
    rotate = TRUE,
    ggtheme = theme_minimal()
)

#### アソシエーション分析 ####
toks <- tokens_compound(toks, pattern = phrase(multiword))

tm <- dfm(toks) %>%
    dfm_remove(tangela::StopWordsJp$word) %>%
    dfm_remove(tangela::ExtendedLettersJp$letter) %>%
    dfm_remove(tangela::OneLettersJp$letter) %>%
    dfm_remove("[0-9]", valuetype = "regex") %>%
    convert(to = "matrix") %>%
    Matrix::Matrix() %>%
    as("realRatingMatrix") %>%
    binarize(minRating = 1) %>%
    as("matrix") %>%
    as("transactions")

rules <- apriori(
    tm,
    parameter = list(
        supp = 0.1,
        maxlen = 7,
        confidence = 0.5
    )
)

inspectDT(rules)

plot(rules)
plot(rules, method = "grouped", measure = "support", shading = "lift", control = list(k = 10))
subrules <- subset(rules, lift > 1.0)
plot(subrules, method = "grouped", measure = "support", shading = "lift", control = list(k = 10))
plot(subrules, method = "graph", measure = "support", shading = "lift")
