dnl -*- Autoconf -*-

AC_DEFUN([AC_CONFIG_WITH_RANGE3],[

AC_MSG_CHECKING([for Boost range version 3])

AS_VAR_PUSHDEF([ac_path_Var],[ac_range3_path])
AS_VAR_PUSHDEF([ac_incdir_Var],[ac_range3_incdir])
AS_VAR_PUSHDEF([ac_cppflags_Var],[ac_range3_cppflags])

dnl
dnl Only one needed if the package tree has a normal structure
dnl
AC_ARG_WITH([range3],
    [AC_HELP_STRING([--with-range3],[Boost range v3 installation directory])],
    [ac_path_Var=${withval}],
    [])

dnl
dnl Override the include directory
dnl
AC_ARG_WITH([range3-incdir],
    [AC_HELP_STRING([--with-range3-incdir],[range3 include directory])],
    [ac_incdir_Var=${withval}],
    [])

if test "x${ac_incdir_Var}" = x; then
  if test "x${ac_path_Var}" != x; then
    ac_cppflags_Var="-I${ac_path_Var}/include"
  fi
else
  ac_cppflags_Var="-I${ac_incdir_Var}"
fi

AC_SUBST(RANGE3_CPPFLAGS,[$ac_cppflags_Var])

AC_MSG_RESULT([done])

AS_VAR_POPDEF([ac_path_Var])
AS_VAR_POPDEF([ac_incdir_Var])
AS_VAR_POPDEF([ac_cppflags_Var])

])
