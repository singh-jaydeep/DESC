# Frozen drivers of designD (written by `../../freeze_drivers.py`)

Each file is `git show <commit>:<path>` for the commit its stage ran at. The repository is canonical; to rerun a stage, check out its commit and run there (see `../REPRODUCE.md`).

* `1_near_axis` @ `4e2a72420`: nc5_s1..s4 (run_na2.py identical to 49557cd7c, the commit in use at launch)
* `2_fb_basis12` @ `da44eb5ec`: Modal fb_nc5lh_b12: score_fb.py --LMN 12 --repair
* `3_single_stage` @ `8aa3ed22f`: Modal ss12_nc5lh_q15b: prep.py + single_stage.py k=1..6
* `4_fb_basis14` @ `29f1c5e36`: Modal fb14_D: fb_rescore.py --LMN 14 --solve-tol 1e-8
* `5_analysis` @ `835590a09`: finalize checks (tracing, interior field, ladders, QS, Gamma_c)

| stage | commit | path | md5 |
|---|---|---|---|
| 1_near_axis | 4e2a72420 | coil_sandbox/from_axis/na/run_na2.py | 4b0356aff5f8458fcfa7d74d088a264b |
| 1_near_axis | 4e2a72420 | coil_sandbox/from_axis/na/nae.py | 615296efac40ae778544561bf6bf1968 |
| 1_near_axis | 4e2a72420 | coil_sandbox/from_axis/na/nae2.py | 2923528354848411ae369aba2eb34f5d |
| 1_near_axis | 4e2a72420 | coil_sandbox/from_axis/na/folded.py | 5e48e9e3f0dc17406da64d44bdacf31b |
| 1_near_axis | 4e2a72420 | coil_sandbox/from_axis/na/cont.py | 366572218e307f7afeacbf700e59db38 |
| 1_near_axis | 4e2a72420 | coil_sandbox/from_axis/na/ncchain.py | 61354e77a42aa86abac37ede061c9367 |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/from_axis/na/score_fb.py | e58cb374bf6d46b094ad9e08a272032d |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/from_axis/na/convert.py | 7312bb289f82fb155155ef260681194c |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/from_axis/na/folded.py | 5e48e9e3f0dc17406da64d44bdacf31b |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/from_axis/na/nae.py | 615296efac40ae778544561bf6bf1968 |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/from_axis/ss/repair_lib.py | 7832d4d6348f7af41209ddb7a6b32f6e |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/from_axis/ss/tip.py | 6cac5b3b6ad8e75deb4539bebea5af23 |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/boot.py | 536136ab3649a9fd19cfc982458cf236 |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/common.py | 6fa6e9e2d07869cefa6347c64256bc83 |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/convexity.py | d23e5d6bed967e3d0b30fdf254b1ab97 |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/planarity.py | b253ae9bb946e98c97415367e3f2677b |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/bdry_repair/README.md | 2b961a97be65980b4745aa3609e1a11e |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/bdry_repair/__init__.py | 5fef0ba4023ceb0b4b36bac93495a110 |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/bdry_repair/_surface_fit_pr2337.py | d026c64a9522d08e3629d05dda0435b5 |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/bdry_repair/core.py | 306e92a7a6a5bf35e93ad0efab3e8e2b |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/bdry_repair/diagnostics.py | 1ac81506eec074c07bb2809a581551ba |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/bdry_repair/interior.py | c38e9bf166bdb5e58572ece62f5e67bb |
| 2_fb_basis12 | da44eb5ec | coil_sandbox/sandbox/bdry_repair/refit.py | ae7fce04acdc84f0b4c151540e6e7130 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/single_stage.py | b1812ccff93ff76b4a16f925a93f5ae3 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/from_axis/ss/prep.py | a34c436681e9069d818655c90707e5ae |
| 3_single_stage | 8aa3ed22f | coil_sandbox/from_axis/ss/fb_rescore.py | 50bb6e0678c9ce5b5cc1f8ae45572024 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/from_axis/ss/tip.py | 6cac5b3b6ad8e75deb4539bebea5af23 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/from_axis/ss/repair_lib.py | 7832d4d6348f7af41209ddb7a6b32f6e |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/boot.py | 536136ab3649a9fd19cfc982458cf236 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/common.py | 6fa6e9e2d07869cefa6347c64256bc83 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/convexity.py | d23e5d6bed967e3d0b30fdf254b1ab97 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/planarity.py | b253ae9bb946e98c97415367e3f2677b |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/bnormal.py | 05546757b106c8c3c0c472ce56b9257f |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/demount.py | ee28cf19866b64c81397f8ef73f13804 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/gauge_fix.py | 0aa7fe19b07e04ddd153ca9ff4170d72 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/qs_scale_invariant.py | 368720e0bbe24723a00eb8e471f28e00 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/subspace_newton.py | d5cf7e229e6c71ec47152384588d1355 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/bounds/fa_qa2_a0167.json | 312c12e0ecb6df24b0a78f578cd03c15 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/from_axis/cloud/app.py | 4f6d766fbc1761e4fc724fe1d22074cf |
| 3_single_stage | 8aa3ed22f | coil_sandbox/from_axis/cloud/jobs.py | 6327cee4aa9b4c5e968119681c84c52d |
| 3_single_stage | 8aa3ed22f | coil_sandbox/from_axis/cloud/requirements.txt | 8276b70f9f39295da5bab53d9b0cbe48 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/from_axis/cloud/watchdog.sh | 0131d7dc09bbefbe23b2d4905b24fb62 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/bdry_repair/README.md | 2b961a97be65980b4745aa3609e1a11e |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/bdry_repair/__init__.py | 5fef0ba4023ceb0b4b36bac93495a110 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/bdry_repair/_surface_fit_pr2337.py | d026c64a9522d08e3629d05dda0435b5 |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/bdry_repair/core.py | 306e92a7a6a5bf35e93ad0efab3e8e2b |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/bdry_repair/diagnostics.py | 1ac81506eec074c07bb2809a581551ba |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/bdry_repair/interior.py | c38e9bf166bdb5e58572ece62f5e67bb |
| 3_single_stage | 8aa3ed22f | coil_sandbox/sandbox/bdry_repair/refit.py | ae7fce04acdc84f0b4c151540e6e7130 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/from_axis/ss/fb_rescore.py | bc9684ba659e30074efeebaa6f290fa8 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/from_axis/ss/tip.py | 6cac5b3b6ad8e75deb4539bebea5af23 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/from_axis/ss/repair_lib.py | 7832d4d6348f7af41209ddb7a6b32f6e |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/boot.py | 536136ab3649a9fd19cfc982458cf236 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/common.py | 6fa6e9e2d07869cefa6347c64256bc83 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/from_axis/cloud/app.py | 4f6d766fbc1761e4fc724fe1d22074cf |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/from_axis/cloud/jobs.py | 12840c54d7d287da30ac9f490d199329 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/from_axis/cloud/requirements.txt | 8276b70f9f39295da5bab53d9b0cbe48 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/bdry_repair/README.md | 2b961a97be65980b4745aa3609e1a11e |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/bdry_repair/__init__.py | 5fef0ba4023ceb0b4b36bac93495a110 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/bdry_repair/_surface_fit_pr2337.py | d026c64a9522d08e3629d05dda0435b5 |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/bdry_repair/core.py | 306e92a7a6a5bf35e93ad0efab3e8e2b |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/bdry_repair/diagnostics.py | 1ac81506eec074c07bb2809a581551ba |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/bdry_repair/interior.py | c38e9bf166bdb5e58572ece62f5e67bb |
| 4_fb_basis14 | 29f1c5e36 | coil_sandbox/sandbox/bdry_repair/refit.py | ae7fce04acdc84f0b4c151540e6e7130 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/iota_trace.py | 7581ad15e9dd7cbab389fc7ee4b35052 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/interior_check.py | c674c6d95f951a51187b435973b0af9a |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/qs_metrics.py | d3a82bf1e8122017c4c7843cfaf5da79 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/qs_booz_fix.py | 58476b9669ae434d53ae006041dc3f23 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/gamma_c.py | 2ec4bb740e8f6751438d86c88e2be2d9 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/resladder.py | b8a8b71575e455d18ed544d0f4c08d7a |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/eps_conv.py | 04ad3ba7deb57482cf6734ae0a1e7d07 |
| 5_analysis | 835590a09 | coil_sandbox/from_axis/ss/z0geom.py | 87327f2817ffecc7504a4c2f27a9c83b |
