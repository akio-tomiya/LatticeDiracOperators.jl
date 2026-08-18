# LatticeDiracOperators v1 棚卸し

更新日: 2026-08-18

## 結論

LatticeDiracOperators 1.0.0 の標準バックエンドを LatticeMatrices に統一する作業は、
Wilson、staggered/HISQ、domain-wall 系まで完了した。v1 の標準 concrete field は
`MPILattice` 系とし、旧 wing/nowing/accelerator 実装は名前と挙動を維持したまま
`deprecated/` に隔離している。

今回の棚卸しで、次の実装上のリリースブロッカーは解消した。

- Wilson--clover の operator/action/fermion force
- Julia 1.12 で LM Wilson operator を組み立てる GeneralFermion callback のAD
- Enzyme を使わない HISQ analytic force
- Enzyme の通常依存から weak dependency / extension への分離
- 公開API、テスト契約、CI matrix、生成物と未使用依存の整理
- QCDMeasurements 1.0.0 からの下流互換確認

未実施なのは、依頼により対象外とした tag・registry・release chain の操作だけである。
ローカルパスでの検証は完了しているが、登録済みバージョンだけを使うclean environment
検証は各依存の公開後に行う。

## 対象バージョン

| package | local version | 用途 |
|---|---:|---|
| LatticeMatrices | 1.1.1 | field、shift、Wilson/clover/staggered/HISQ/domain-wall kernel |
| Gaugefields | 1.1.0 | LatticeMatrices を既定storageとする gauge wrapper |
| LatticeDiracOperators | 1.0.0 | 本作業ツリー |
| QCDMeasurements | 1.0.0 | downstream integration |

LDO の `Project.toml` は `Gaugefields = "1"`、`LatticeMatrices = "1.1"`、
`julia = "1.11"` を要求する。Enzyme 0.13 は weak dependency であり、
通常ロードには不要である。

## v1 の標準経路

| fermion | 標準 field | LatticeMatrices backend |
|---|---|---|
| Wilson / Wilson--clover | `WilsonFermion_4D_MPILattice` | `WilsonDiracOperator4D` / `WilsonDiracCloverOperator4D` |
| staggered | `StaggeredFermion_4D_MPILattice` | `StaggeredDiracOperator4D` |
| HISQ | `StaggeredFermion_4D_MPILattice` | `HISQDiracCache4D` と analytic link pullback |
| domain-wall | `DomainwallFermion_5D_MPILattice` | LM 5D operator |
| generalized domain-wall | `DomainwallFermion_5D_MPILattice` | LM generalized 5D operator |
| Möbius domain-wall | MPILattice 標準経路 | LM Möbius 5D operator |

旧 concrete type、演算子、kernel は各 fermion ディレクトリの `deprecated/` へ移した。
従来ファイルを移動して include を直した構成なので、関数名・型名による後方互換経路は
残っている。標準コードからは deprecated 実装を選ばない。

## Wilson

### 通常 Wilson

`WilsonFermion_4D_MPILattice.jl` と `linearalgebra_4D.jl` を
`mpi_jacc/` から標準階層へ移した。LM と重複する `axpby!`、scalar `mul!`、
`Oneγμ` 等は active 実装を止め、LM 実装へ委譲した。LM に同値APIがない
projector、outer product、補助演算だけをLDO側に残している。

### Wilson--clover

`Dirac_operator(..., "WilsonClover")` という公開入口を維持しつつ、
内部は LM の `WilsonDiracCloverOperator4D` を包む。forward/adjoint、cache refresh、
D†D、再構築、境界条件は通常Wilson wrapperと同じ契約を使う。

fermion force は Enzyme extension から LM の
`mul_cached_clover!` custom ruleを利用する。有限差分、1 rank、2 rank MPI の
比較を通している。Enzymeをロードしない場合、operatorのforward/adjointは使えるが、
clover forceは必要条件を説明する明示errorを返す。

## Staggered と HISQ

staggered の標準 operator は LM の `StaggeredDiracOperator4D` wrapperである。
forward/adjoint/D†D、action、forceを旧per-site参照実装と比較している。

HISQ は staggered action の一種として `src/StaggeredFermion/` に置いた。
fat/long link cacheとoperatorは LM に委譲し、LDOはactionとgauge wrapperへの変換を担う。
forceは LM の `hisq_link_pullback!` を使う解析的 reverse passであり、
Enzymeなしで動く。Enzyme版 GeneralFermionAction の例も独立に残している。

## Domain-wall

Shamir、Möbius、generalized domain-wall の MPILattice field/operatorを標準階層へ移し、
旧実装を `deprecated/` に隔離した。LM backendとのforward/adjoint/D†D、
action、force、非自明なslice係数を検証している。

## GeneralFermion

`GeneralFermionAction` は、ユーザー定義の `apply_D` callbackをそのままADできる経路として
残している。examplesには次を追加した。

- `GeneralFermion_Quickstart.jl`: 最小の `apply_D` 定義
- `GeneralFermion_Shift_AD.jl`: shiftでstencilを組むAD例
- `GeneralFermion_AllDirections.jl`: `U1`、`U2`、`U3`、`U4` をすべて使う例

LM Wilson callback用には、AD-safeな
`WilsonDiracOperator4D(U1, U2, U3, U4, κ)` constructorをLMに追加した。
Julia 1.12でvector literalをcallback内に作った際のEnzyme ABI/type解析問題を避ける。
`operator.κ` の公開挙動はFloat64のまま維持している。

## 公開API

v1で標準とするMPILattice型をトップレベルからexportした。
`cg` と `WilsonFermion_4D_wing` は既存exportが未定義だったため、互換moduleから
正しくimportするよう修正した。全exportが定義済みであることを `test/public_api.jl`
で検査する。

公開表面とdeprecated境界は `docs/src/v1_api.md` に記録した。

## Enzyme境界

- Enzymeはweak dependencyで、`LatticeDiracOperatorsEnzymeExt` からのみ追加機能をロードする。
- 通常のWilson/staggered/HISQ/domain-wall operatorはEnzymeなしでロード・実行できる。
- HISQ forceはLMのanalytic pullback。
- Wilson--clover forceはLMのanalytic pullbackでEnzymeなし。
- 任意callbackの自動微分はEnzymeあり。
- LM、Gaugefields、LDOの各extensionが同一メソッドを上書きしないよう責務を分離した。

## Test / CI

テスト入口を用途別に分けた。

- `test/runtests_core.jl`: Enzymeをロードしないcore
- `test/runtests_ad.jl`: GeneralFermion、LM callback、HISQ AD
- `test/runtests_mpi.jl`: 2-rank MPI
- `test/runtests.jl`: Julia 1.11のlegacyを含むfull package suite

重いJACC/Enzymeテストは、world age、method overwrite、GC量の相互干渉を避けるため
分離runner内でファイルごとのJulia subprocessとして実行する。hot gaugeを使う
有限差分テストは乱数seedを固定し、実行順に依存しない。

CIは次を実行する。

- core/no-Enzyme: Julia 1.11・1.12 Linux、1.11 macOS/Windows
- Enzyme: Julia 1.11・1.12 Linux
- 2-rank MPI: Julia 1.11・1.12 Linux
- legacyを含むfull package test: Julia 1.11 Linux

### ローカル検証結果

| 検証 | 結果 |
|---|---|
| Julia 1.11 full package suite（README整理前、source同一） | 409/409、約10分08秒 |
| README掲載blockの直接実行、Julia 1.11/1.12 | 7/7 each、Enzyme未load |
| README test統合後のJulia 1.12 core/no-Enzyme runner | 10/10 subprocess pass |
| Julia 1.11 analytic Wilson--clover | LM finite difference pass、LDO action 19/19、2-rank 19/19 per rank |
| Julia 1.11 LM clover Enzyme compatibility | direct 4/4、cached explicit-link 12/12 |
| Julia 1.12 Enzyme components | LM callback 3/3、HISQ HMC 4/4、MPIJACC AD pass |
| Julia 1.12 2-rank MPI | halo、AD、clover、callback、staggered、HISQ、domain-wall 全pass |
| LM Wilson AD smoke, Julia 1.11/1.12 | 25/25 each |
| QCDMeasurements 1.0.0 CPU downstream, Julia 1.11 | 175/175 |

## 整理

- runtimeで未使用だった `Optimisers` を依存関係から削除した。
- `InteractiveUtils` はlegacy testだけが使うためtest extraへ移した。
- source内の未使用 `using InteractiveUtils` を削除した。
- `.DS_Store`、package/docs Manifest、LocalPreferences、ログ、出力、
  docs build、editor backup、未参照の `test/filelist.dat` をGit管理対象から削除した。
- `.gitignore` に上記生成物を追加し、run directory全体を誤ってignoreする指定を外した。
- main READMEをGaugefields v1と同じく推奨API中心の短い構成へ整理し、掲載する
  Julia code blockをすべてREADMEから抽出してJulia 1.11/1.12で実行するtestへ変更した。

削除したGit管理ファイルは履歴から復元できる。

## 残件

### 今回の対象外

- LatticeMatrices、Gaugefields、LatticeDiracOperators、QCDMeasurementsのtag作成
- General registryへの登録
- 登録済みversionだけを使うclean environment CI

### リリース後にも継続できる改善

- deprecated APIの削除時期とdeprecation warningの方針を別途決める。
- legacy HMC testが同名helperを上書きする警告を、test module分離で解消する。
- CUDA実機の性能・allocation回帰をCI外の定期benchmarkとして整備する。

## 判定チェックリスト

- [x] LMを標準backendとするMPILattice実装
- [x] Wilson、clover、staggered、HISQ、domain-wallの数値回帰
- [x] HISQ analytic force（Enzymeなし）
- [x] Wilson--clover force有限差分
- [x] GeneralFermion callback AD on Julia 1.11/1.12
- [x] public export smoke test
- [x] no-Enzyme / Enzyme / 2-rank MPI CI定義
- [x] QCDMeasurements v1 downstream test
- [x] generated filesと未使用runtime dependencyの整理
- [ ] published dependencyだけを使うrelease CI（今回対象外）
