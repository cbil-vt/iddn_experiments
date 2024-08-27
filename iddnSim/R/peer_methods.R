library(JGLtry)


jgl_wrap <- function(n, dat1_lst, dat2_lst, l1_rg, l2_rg, n_sample_work, sigma_add) {
  dat1 = dat1_lst[n,,]
  dat2 = dat2_lst[n,,]

  # n_sample = dim(dat1)[1]
  n_feature = dim(dat1)[2]
  n_l1 = length(l1_rg)
  n_l2 = length(l2_rg)

  # lambda1, lambda2, conditions, feature, feature
  dep_est_arr0 = array(NA, c(n_l1, n_l2, 2, n_feature, n_feature))

  # idx1 = sample(n_sample, n_sample_work, replace = FALSE)
  # idx2 = sample(n_sample, n_sample_work, replace = FALSE)

  # Must scale data first
  dat1_sel = dat1[1:n_sample_work, 1:n_feature]
  dat2_sel = dat2[1:n_sample_work, 1:n_feature]
  dat1_sel = dat1_sel + array(rnorm(n_sample_work*n_feature)*sigma_add, c(n_sample_work,n_feature))
  dat2_sel = dat2_sel + array(rnorm(n_sample_work*n_feature)*sigma_add, c(n_sample_work,n_feature))

  dat1_sel = scale(dat1_sel)
  dat2_sel = scale(dat2_sel)
  dat_lst = list(dat1_sel, dat2_sel)

  for (i in 1:n_l1) {
    for (j in 1:n_l2) {
      l1 = l1_rg[i]
      l2 = l2_rg[j]
      print(paste(n, l1, l2))
      tryCatch({
        out = JGL(
          dat_lst,
          lambda1 = l1,
          lambda2 = l2,
          return.whole.theta = TRUE
        )
        x1 = out$theta[[1]]
        x2 = out$theta[[2]]
        dep_est_arr0[i, j, 1, , ] = x1
        dep_est_arr0[i, j, 2, , ] = x2
      }, error = function(err) {
        # error handler picks up where error was generated
        print(paste("MY_ERROR:  ",err))
        print(dat1_sel)
        print(dat2_sel)
        idx = sample.int(100000, 1)
        out = list(dat1_sel=dat1_sel, dat2_sel=dat2_sel, l1=l1, l2=l2)
        saveRDS(out, paste0("jgl_bad_", idx, ".rds"))
      })
    }
  }
  return(dep_est_arr0)
}


collect_parallel_res <- function(res_lst) {
  # Repeat, lambda1, lambda2, conditions, feature, feature
  n_rep = length(res_lst)
  d_all = c(n_rep, dim(res_lst[[1]]))
  dep_est_arr = array(0.0, d_all)
  for (n in 1:n_rep) {
    dep_est_arr[n,,,,,] = res_lst[[n]]
  }
  return(dep_est_arr)
}

