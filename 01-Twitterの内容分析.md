# 1. Twitterの内容分析

## このセクションでやること

斗和キセキはとあるVtuberですが、ちょうどこの本を書いている時期に、背中の意匠が「レッドフレーム改」と呼ばれるガンダムに似ているということでにわかに話題になりました。

[【斗和キセキ】Vtuber、背中のパーツがレッドフレーム改っぽいというだけでフォロワー数＆チャンネル登録が激増する - Togetter](https://togetter.com/li/1325312)

その後、何気なくつぶやかれた挨拶から今度は仮面ライダーの登場人物（「エボルト」）に結び付けられて話題になるというよくわからない顛末をたどっています。

このセクションでは{rtweet}で「レッドフレーム改」を含むツイートを取得し、内容分析して、斗和キセキがどのように話題にされているかを確認します。

## {tangela}のインストール

以下を実行すると入ります。

```R
devtools::install_github("paithiov909/tangela")
tangela::npm_install()
```

また、このセクションではkuromoji.js用のNEologd辞書を使用しています。kuromoji.jsには辞書ファイルが同梱されているので必須ではありませんが、使いたい場合は、以下のリポジトリのコードを利用して辞書ファイルを生成するとよいでしょう。

[frontainer/kuromoji-js-dictionary: kuromoji.js dictionary generator](https://github.com/frontainer/kuromoji-js-dictionary)

`tangela::tokenize()`と`tangela::documentTermMatrix()`を使用するためには、あらかじめ`tangela::start_tangela()`を実行してExpressサーバーを立ち上げて置く必要があります。もし生成した辞書ファイルを使用したい場合には、[dotenv](https://www.npmjs.com/package/dotenv)のコンフィグファイル内でDICPATHをたとえば以下のように指定します。

```dotenv.env
HOST=127.0.0.8
PORT=3033
DICPATH=C:\Users\hogehoge\Documents\zatsuniyaru\R\dist
CLUSTER=2
```

dotenvのコンフィグファイルは`tangela::start_tangela(dotenv = file.path("./dotenv.env"))`のようなかたちで`start_tangela()`に渡してください。以降、この本のなかでは上記のdotenvコンフィグファイルを渡して`start_tangela()`を実行していることを想定します。

## その他のパッケージ

{tangela}のほかに次のパッケージを使うので必要に応じてインストールしましょう。

```R
library(tidyverse)
library(feather)
library(rtweet)
library(quanteda)
library(ggpubr)
```

なお、{rtweet}については現在は開発用アカウントを取らなくても使えるようになっています。[Githubリポジトリ](https://github.com/mkearney/rtweet#api-authorization)に次のように書かれているので確認してください。

> **It is no longer necessary to obtain a developer account and create your own Twitter application to use Twitter’s API.** You may still choose to do this (gives you more stability and permissions), but **{rtweet}** should work out of the box assuming (a) you are working in an interactive/live session of R and (b) you have installed the [**{httpuv}**](https://github.com/rstudio/httpuv) package.

## データの準備

`rtweet::search_tweets()`でツイートを取得します。Twitter APIの仕様でクエリによっては指定した件数を取得できないことがありますが、ここでは大して問題ないので無視します。不完全ですが一応、NEologdのリポジトリが推奨している文字列の正規化処理をおこなっています。おこなうべき前処理については[こちら](https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja)を参照してください。

```R
rt <- search_tweets(
    "レッドフレーム改",
    n = 2000,
    include_rts = FALSE,
    retryonratelimit = TRUE
) %>%
    select(created_at, text) %>%
    write_feather("tweets.feather")

rt$text <- rt$text %>%
    str_replace_all(regex("[’]"), "\'") %>%
    str_replace_all(regex("[”]"), "\"") %>%
    str_replace_all(regex("[˗֊‐‑‒–⁃⁻₋−]+"), "-") %>%
    str_replace_all(regex("[﹣－ｰ—―─━ー]+"), "ー") %>%
    str_remove_all(regex("[~∼∾〜〰～]")) %>%
    str_remove_all("[:punct:]") %>%
    str_remove_all("[:blank:]") %>%
    str_remove_all("[:cntrl:]")
```

次に、取得したデータを{tangela}で分かち書きにします。行ごとに`tangela::tokenize()`に渡すよりも、分かち書きしたいテキストをリストにして渡してしまって戻り値を加工するほうが速いので、そのようにします。

```R
rt$tokens <- tangela::tokenize(
    docs = as.list(rt$text),
    host = "127.0.0.8",
    port = 3033
) %>%
    map(~ map(., ~ .x$surface_form)) %>%
    map(~ paste(., collapse = " ")) %>%
    purrr::flatten() %>%
    unlist()
```

分かち書きしたテキストを含むデータフレームを{quanteda}のコーパスに格納します。

```R
corp <- rt %>%
    distinct(tokens, .keep_all = TRUE) %>%
    corpus(text_field = "tokens")
```

## 内容分析

NEologd辞書には「レッドフレーム」という語彙項目が存在しているので、分かち書きした段階で「レッドフレーム」という単語を検出できているはずです。`quanteda::kwic()`で「レッドフレーム」の前後の文脈を確認してみます。

```R
toks <- tokens(corp, what = "fastestword", remove_punct = TRUE, remove_twitter = TRUE)

kwic(toks, "レッドフレーム") %>%
    as_tibble() %>%
    View()
```



![kwic_redframe](http://t29.pixhost.to/thumbs/352/101284950_kwic_redframe.png)

「レッドフレーム」と「改」が2つの単語にわかれているのがわかります。コロケーションを集計してみましょう。

```R
textstat_collocations(toks, size = 2) %>%
    as_tibble() %>%
    View()
```

![collocation](http://t29.pixhost.to/thumbs/352/101284947_collocation.png)

コロケーションはzの降順に並べています（zが何か知りたい人は{quanteda}のマニュアルを読みましょう）。ここでは「レッドフレーム改」や「斗和キセキ」などは一語として扱ってもよさそうなのでまとめて扱うことにします。

```R
multiword <- c("レッドフレーム 改", "斗和 キセキ")
toks <- tokens_compound(toks, pattern = phrase(multiword))
```

これで「レッドフレーム改」を「レッドフレーム\_改」、「斗和キセキ」を「斗和\_キセキ」という文字列で抽出できるようになっています。あらためて`quanteda::kwic()`で前後の文脈を見ると次のようになります。

```R
kwic(toks, "レッドフレーム_改") %>%
    as_tibble() %>%
    View()
```

![kwic_comp_redframe](https://t29.pixhost.to/thumbs/352/101284949_kwic_comp_redframe.png)

コロケーションもあらためて集計してみます。

```R
textstat_collocations(toks, size = 2) %>%
    as_tibble() %>%
    View()
```

![collocation_comp](https://t29.pixhost.to/thumbs/352/101284946_collacation_comp.png)

しかし、このコロケーションの集計からはいまいち何が起きているのかわかりません。確認する共起関係のサイズをすこし大きくして、3-gramのワードクラウドを描いてみることにします。

```R
tokens(corp, what = "fastestword", remove_punct = TRUE, remove_twitter = TRUE) %>%
    tokens_compound(pattern = phrase(multiword)) %>%
    tokens_skipgrams(n = 3, skip = 0) %>%
    dfm() %>%
    textplot_wordcloud(
        min_count = 10,
        random_order = FALSE,
        color = viridisLite::cividis(8)
    )
```

![word_cloud](https://t29.pixhost.to/thumbs/352/101284961_wordcloud.png)

ごちゃごちゃしていますが、「レッドフレーム改の次はエボルトになった」という話の片鱗が見えてきました。今度は3-gramの共起ネットワークを描いてみます。

```R
tokens(corp, what = "fastestword", remove_punct = TRUE, remove_twitter = TRUE) %>%
    tokens_compound(pattern = phrase(multiword)) %>%
    tokens_skipgrams(n = 3, skip = 0) %>%
    dfm() %>%
    fcm() %>%
    fcm_select(., names(topfeatures(., 150))) %>%
    textplot_network(min_freq = 0.95, edge_size = 5)
```

![network](http://t29.pixhost./thumbs/w/352/101284951_network.png)

「斗和キセキがレッドフレーム改の次はエボルト扱いされている」らしいことがなんとなく見えてきた気がします。単純な頻度とはべつな角度からこうした共起関係が多く見られることを確認するために、アソシエーション分析をしてみましょう。

ひとつのツイート内に一度でも出現した単語をOne-hotでカウントするために、一度{recommenderlab}のrealRatingMatrixクラスのオブジェクトに変換して`binarize()`してから、{arules}のtransactionsクラスのオブジェクトに変換します。

```R
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
```

`arules::apriori()`を実行します。

```R
rules <- apriori(
    tm,
    parameter = list(
        supp = 0.1,
        maxlen = 7,
        confidence = 0.5
    )
)

inspectDT(rules)
```

[このように](https://paithiov909.github.io/zatsuniyaru/)「レッドフレーム改な斗和キセキがエボルトである」らしいことが確認できます。

![inspect](https://t29.pixhost.to/thumbs/352/101284948_inspect.png)

## まとめ

このセクションでは{rtweet}で「レッドフレーム改」を含むツイートを取得し、内容分析して、斗和キセキがどのように話題にされているかを確認してきました。テキストマイニングの文脈でアソシエーション分析が紹介されることはあまりないように思いますが、頻度情報に注目して内容分析をしようとすると、全文書を通じてのカウントはすくなくてもconfidenceの高い共起関係（アソシエーションルール）を見つけるのが難しかったりします。埋もれがちだけれども注目する意味のある共起関係を見つけ出すひとつの手がかりとして、アソシエーション分析のような手法も上手く活用したいものです。