/* qpstage.h - the stage-wise factorization of the QP interior-point method's system (qpstage.c; the QP developer,
 * 14 November 2026). Internal to the QP solver. */
#ifndef BRISK_QPSTAGE_H
#define BRISK_QPSTAGE_H
#include <stddef.h>
/* the sum of a_i b_i, and a += t c, over m entries (with AVX-512: eight a step, the last step masked) */
double qp_vscale_max(int m, double *a, double fj, const double *fr, double *rmax, double cmax);
double qp_vdot(int m, const double *a, const double *b);
void qp_vaxpy(int m, double *a, const double *c, double t);
typedef struct QpStage QpStage;
/* the levels of the problem (n variables, m rows, H's lower triangle and A by columns, eq: the equality rows) and the
 * blocks' memory; NULL when there is no variable, when the blocks would hold more than cap entries, or without memory */
QpStage *qp_stage_build(int n, int m, const int *Hp, const int *Hi, const int *Ap, const int *Ai, const char *eq, size_t cap);
/* the blocks filled and factored for d (n: added to H's diagonal) and e (m: the rows' diagonal); 0, or 1 on a block that
 * is not positive definite */
int qp_stage_factor(QpStage *S, const int *Hp, const double *Hx, const int *Ap, const double *Ax, const double *d, const double *e);
/* [-(H + diag d) A'; A diag e] (dx; dy) = b solved in place on b (n + m) */
void qp_stage_solve(QpStage *S, const double *Ax, double *b);
void qp_stage_free(QpStage *S);
int qp_stage_levels(const QpStage *S);          /* the number of levels */
int qp_stage_widest(const QpStage *S);          /* the widest block (variables of a level, or equality rows of a block) */
double qp_stage_fact_work(const QpStage *S);    /* operations of a factorization and of a solve, from the blocks' sizes */
double qp_stage_solve_work(const QpStage *S);
size_t qp_stage_entries(const QpStage *S);      /* entries of the blocks */

/* the normal equations by stage (H diagonal): M = A diag(w) A' + diag(e), block tridiagonal over the rows' levels */
typedef struct QpStageNe QpStageNe;
/* own > 0: the automatic rule's conditions on the levels' sizes are applied before the blocks are allocated (own: the
 * augmented system's entries before any fill); NULL when one fails, with info[0..3] (may be NULL) = the factor's entries,
 * the blocks' width by work, the factorization's work, the forming's work */
QpStageNe *qp_stagene_build(int n, int m, const int *Ap, const int *Ai, size_t cap, double own, double *info);
int qp_stagene_factor(QpStageNe *S, const int *Ap, const int *Ai, const double *Ax, const double *w, const double *e);
void qp_stagene_solve(QpStageNe *S, double *y);      /* y := M^-1 y */
void qp_stagene_free(QpStageNe *S);
int qp_stagene_levels(const QpStageNe *S);
int qp_stagene_widest(const QpStageNe *S);
double qp_stagene_fact_work(const QpStageNe *S);
double qp_stagene_solve_work(const QpStageNe *S);
double qp_stagene_form_work(const QpStageNe *S);
/* how dense M is inside its diagonal blocks, by factorization work (1: full blocks, work no ordering avoids; near 0:
 * sparse inside wide levels) */
double qp_stagene_dense(const QpStageNe *S);
double qp_stagene_lnz(const QpStageNe *S);
double qp_stagene_width(const QpStageNe *S);    /* the blocks' width weighted by their work (sum r^3 / sum r^2) */      /* entries of the factor (the blocks' lower triangles and the coupling blocks) */
size_t qp_stagene_entries(const QpStageNe *S);
#endif
