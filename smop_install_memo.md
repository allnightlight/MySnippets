# はじめに

[SMOP](https://github.com/victorlei/smop)をインストールした。
留意点を記録する。

# インストールの手順

1. コマンドプロントを開く
2. `pip install smop`でsmopをインストールする
3. `pip install networkx==1.11`でnetworkxをインストールする
  * 最新バージョンのnetworkxでは動かない
  * バージョンを指定してインストールすること
  * 詳細は[こちら参照](https://github.com/victorlei/smop/issues/165)

# 動作確認

1. 以下のファイルを作成する。ファイル名は、`stat.m`とする。


    function [m,s] = stat(x)
        n = length(x);
        m = sum(x)/n;
        s = sqrt(sum((x-m).^2/n));
    end
2. コマンドプロントを開いて以下のコマンドを実行する。
> smop stat.m
3. 以下のファイルが生成されることを確認する。


    # Generated with SMOP  0.41
    from libsmop import *
    # stat.m


    @function
    def stat(x=None,*args,**kwargs):
        varargin = stat.varargin
        nargin = stat.nargin

        n=length(x)
    # stat.m:2
        m=sum(x) / n
    # stat.m:3
        s=sqrt(sum((x - m) ** 2 / n))
    # stat.m:4
        return m,s

    if __name__ == '__main__':
        pass
