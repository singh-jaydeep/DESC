# Frozen drivers of designB (written by `../../freeze_drivers.py`)

Each file is `git show <commit>:<path>` for the commit its stage ran at. The repository is canonical; to rerun a stage, check out its commit and run there (see `../REPRODUCE.md`).

* `0_near_axis_reference` @ `846ece7cb`: reference lineage s1 -> o2w1 (ran from uncommitted code; 846ece7cb is the closest commit; s1 has no exact commit)
* `1_near_axis` @ `e5b0a9ad2`: li_qa2_iota0.30 (ran uncommitted; e5b0a9ad2 adds only --wplanar, default 0)
* `2_fb_basis12` @ `a322c7f22`: laptop L12 shells 2,4,6 and the resumed shell 8 (a322c7f22 = the run's working copy as --chunk 200)
* `2b_repair_k6` @ `89f5a39e4`: old equal-arclength refit of shell 6 between the two L12 runs
* `3_single_stage` @ `6993369dc`: Modal ss12_li030_q15: prep.py + single_stage.py k=1..6 (old repair_lib)
* `3b_repair` @ `d5c8966e2`: laptop: li030/fb_norepair_local (fb_rescore.py --shells 4,8 at 6993369dc, stage 3) repaired with the new bdry_repair package -> li030/repaired/eq.h5 (ran uncommitted; committed 5 min later as d5c8966e2)
* `4_basis14_and_polish` @ `835590a09`: laptop: basis-14 re-solve, free-boundary shell, coil polish (run_al.py), free-boundary shell (code unchanged from the run times to this commit)
* `5_analysis` @ `835590a09`: finalize checks (tracing, interior field, ladders, QS, Gamma_c)

| stage | commit | path | md5 |
|---|---|---|---|
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/run_na2.py | 4663bd24b35d55a14a2caea9048fef5d |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/nae.py | 615296efac40ae778544561bf6bf1968 |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/nae2.py | 2923528354848411ae369aba2eb34f5d |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/folded.py | 5e48e9e3f0dc17406da64d44bdacf31b |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/cont.py | 366572218e307f7afeacbf700e59db38 |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/sweep1.sh | 27c5537eda0bd9c5db9e1ad73ba45f24 |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/reg1.sh | 90fc9c141723e62f2325df4858ed4e6d |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/reg2.sh | a5403977f3ef402b79f0a94ac6fddca6 |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/o2.sh | 7cd60ef05ff3983583542460a5a0de13 |
| 0_near_axis_reference | 846ece7cb | coil_sandbox/from_axis/na/w1test.sh | 0321a0c6a220bbc2b5eee342ea9b5be8 |
| 1_near_axis | e5b0a9ad2 | coil_sandbox/from_axis/na/run_na2.py | ed809d0b6bfa8c842fe228a21527bd0b |
| 1_near_axis | e5b0a9ad2 | coil_sandbox/from_axis/na/nae.py | 615296efac40ae778544561bf6bf1968 |
| 1_near_axis | e5b0a9ad2 | coil_sandbox/from_axis/na/nae2.py | 2923528354848411ae369aba2eb34f5d |
| 1_near_axis | e5b0a9ad2 | coil_sandbox/from_axis/na/folded.py | 5e48e9e3f0dc17406da64d44bdacf31b |
| 1_near_axis | e5b0a9ad2 | coil_sandbox/from_axis/na/cont.py | 366572218e307f7afeacbf700e59db38 |
| 1_near_axis | e5b0a9ad2 | coil_sandbox/from_axis/na/lowiota.sh | 103d7c7e8cb28e0889e48d769253fa41 |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/from_axis/na/score_fb.py | d89f0f3a2e3ff5dc53febe822c5d17d5 |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/from_axis/na/convert.py | 7312bb289f82fb155155ef260681194c |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/from_axis/na/folded.py | 5e48e9e3f0dc17406da64d44bdacf31b |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/from_axis/na/nae.py | 615296efac40ae778544561bf6bf1968 |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/from_axis/ss/repair_lib.py | NOT IN THIS COMMIT |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/from_axis/ss/tip.py | NOT IN THIS COMMIT |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/sandbox/boot.py | 536136ab3649a9fd19cfc982458cf236 |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/sandbox/common.py | 6fa6e9e2d07869cefa6347c64256bc83 |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/sandbox/convexity.py | d23e5d6bed967e3d0b30fdf254b1ab97 |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/sandbox/planarity.py | b253ae9bb946e98c97415367e3f2677b |
| 2_fb_basis12 | a322c7f22 | coil_sandbox/from_axis/na/fbscreen.sh | a6f27fac328f81e2e6cb8299ebab31da |
| 2b_repair_k6 | 89f5a39e4 | coil_sandbox/from_axis/ss/repair_eq.py | b99420b6af1033f25986b7c1d7aae663 |
| 2b_repair_k6 | 89f5a39e4 | coil_sandbox/from_axis/ss/repair_lib.py | 5cea165b2436cc3ae8299f75fd10cdf5 |
| 2b_repair_k6 | 89f5a39e4 | coil_sandbox/sandbox/boot.py | 536136ab3649a9fd19cfc982458cf236 |
| 2b_repair_k6 | 89f5a39e4 | coil_sandbox/sandbox/common.py | 6fa6e9e2d07869cefa6347c64256bc83 |
| 3_single_stage | 6993369dc | coil_sandbox/single_stage.py | c0f18ecfc688e4c42fa4bcdc0c6934c7 |
| 3_single_stage | 6993369dc | coil_sandbox/from_axis/ss/prep.py | a34c436681e9069d818655c90707e5ae |
| 3_single_stage | 6993369dc | coil_sandbox/from_axis/ss/fb_rescore.py | cddb9086769f2b52f69ee93c50678663 |
| 3_single_stage | 6993369dc | coil_sandbox/from_axis/ss/tip.py | 6cac5b3b6ad8e75deb4539bebea5af23 |
| 3_single_stage | 6993369dc | coil_sandbox/from_axis/ss/repair_lib.py | 5cea165b2436cc3ae8299f75fd10cdf5 |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/boot.py | 536136ab3649a9fd19cfc982458cf236 |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/common.py | 6fa6e9e2d07869cefa6347c64256bc83 |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/convexity.py | d23e5d6bed967e3d0b30fdf254b1ab97 |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/planarity.py | b253ae9bb946e98c97415367e3f2677b |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/bnormal.py | 05546757b106c8c3c0c472ce56b9257f |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/demount.py | d1d065f2fb6410d7d538898d9c718c83 |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/gauge_fix.py | 0aa7fe19b07e04ddd153ca9ff4170d72 |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/qs_scale_invariant.py | 368720e0bbe24723a00eb8e471f28e00 |
| 3_single_stage | 6993369dc | coil_sandbox/sandbox/subspace_newton.py | d5cf7e229e6c71ec47152384588d1355 |
| 3_single_stage | 6993369dc | coil_sandbox/bounds/fa_qa2_a0167.json | 312c12e0ecb6df24b0a78f578cd03c15 |
| 3_single_stage | 6993369dc | coil_sandbox/from_axis/cloud/app.py | 4f6d766fbc1761e4fc724fe1d22074cf |
| 3_single_stage | 6993369dc | coil_sandbox/from_axis/cloud/jobs.py | 1109c8fe08804a09cd513fdfe4f6fe75 |
| 3_single_stage | 6993369dc | coil_sandbox/from_axis/cloud/requirements.txt | 8276b70f9f39295da5bab53d9b0cbe48 |
| 3_single_stage | 6993369dc | coil_sandbox/from_axis/cloud/watchdog.sh | NOT IN THIS COMMIT |
| 3b_repair | d5c8966e2 | coil_sandbox/from_axis/ss/repair_eq.py | a989fa660513d483f03fe347e97ef94d |
| 3b_repair | d5c8966e2 | coil_sandbox/from_axis/ss/repair_lib.py | 7832d4d6348f7af41209ddb7a6b32f6e |
| 3b_repair | d5c8966e2 | coil_sandbox/sandbox/boot.py | 536136ab3649a9fd19cfc982458cf236 |
| 3b_repair | d5c8966e2 | coil_sandbox/sandbox/bdry_repair/README.md | 2b961a97be65980b4745aa3609e1a11e |
| 3b_repair | d5c8966e2 | coil_sandbox/sandbox/bdry_repair/__init__.py | 5fef0ba4023ceb0b4b36bac93495a110 |
| 3b_repair | d5c8966e2 | coil_sandbox/sandbox/bdry_repair/_surface_fit_pr2337.py | d026c64a9522d08e3629d05dda0435b5 |
| 3b_repair | d5c8966e2 | coil_sandbox/sandbox/bdry_repair/core.py | 306e92a7a6a5bf35e93ad0efab3e8e2b |
| 3b_repair | d5c8966e2 | coil_sandbox/sandbox/bdry_repair/diagnostics.py | 1ac81506eec074c07bb2809a581551ba |
| 3b_repair | d5c8966e2 | coil_sandbox/sandbox/bdry_repair/interior.py | c38e9bf166bdb5e58572ece62f5e67bb |
| 3b_repair | d5c8966e2 | coil_sandbox/sandbox/bdry_repair/refit.py | ae7fce04acdc84f0b4c151540e6e7130 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/from_axis/ss/resladder.py | b8a8b71575e455d18ed544d0f4c08d7a |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/from_axis/ss/fb_rescore.py | bc9684ba659e30074efeebaa6f290fa8 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/from_axis/ss/repair_lib.py | 7832d4d6348f7af41209ddb7a6b32f6e |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/run_al.py | c3c96539428893dca7ee773a8362768a |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/sandbox/al_problem.py | 2d9f8ede19830e2fa344779f031cc214 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/sandbox/boot.py | 536136ab3649a9fd19cfc982458cf236 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/sandbox/common.py | 6fa6e9e2d07869cefa6347c64256bc83 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/sandbox/demount.py | ee28cf19866b64c81397f8ef73f13804 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/sandbox/convexity.py | d23e5d6bed967e3d0b30fdf254b1ab97 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/sandbox/planarity.py | b253ae9bb946e98c97415367e3f2677b |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/check.py | 17cc4824abdfe3a370d16f5387b5b552 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/bounds/fa_qa2_a0167.json | 312c12e0ecb6df24b0a78f578cd03c15 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/from_axis/final_work/res/polish_Bk6/bounds_backoff.json | 25e215d4f8c28dcb66119103ce3eb1c2 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/from_axis/final_work/res/run_B.sh | f56589554cbb07923997ca1241fc9238 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/from_axis/final_work/res/overnight_B.sh | 9b0ade4627ca60a52af3be44c3481291 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/from_axis/final_work/res/replan_B.sh | fdfe80a320a1a2bf2633bef574f5db97 |
| 4_basis14_and_polish | 835590a09 | coil_sandbox/from_axis/final_work/res/polish_B.sh | 21d1a1fc3f157f00bd755a05e186cd1e |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/iota_trace.py | 7581ad15e9dd7cbab389fc7ee4b35052 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/interior_check.py | c674c6d95f951a51187b435973b0af9a |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/qs_metrics.py | d3a82bf1e8122017c4c7843cfaf5da79 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/qs_booz_fix.py | 58476b9669ae434d53ae006041dc3f23 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/gamma_c.py | 2ec4bb740e8f6751438d86c88e2be2d9 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/resladder.py | b8a8b71575e455d18ed544d0f4c08d7a |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/eps_conv.py | 04ad3ba7deb57482cf6734ae0a1e7d07 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/z0geom.py | 87327f2817ffecc7504a4c2f27a9c83b |
