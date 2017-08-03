dnl -*- Autoconf -*-

AC_DEFUN([AC_CONFIG_CXX_WARNINGS],[

CLANG_WARNFLAGS=" \
-Wno-deprecated-register \
-Wno-logical-op-parentheses"

GCC_WARNFLAGS=" -W -Wall \
-Wno-parentheses"

test -z "$OSNAME" && OSNAME=$( uname )

case $CXX in
    *clang++)
        CXXFLAGS+=$GCC_WARNFLAGS
        ;;
    *g++)
        if test "$OSNAME" = "Darwin"; then
            CXXFLAGS+=$CLANG_WARNFLAGS
        else
            CXXFLAGS+=$GCC_WARNFLAGS
        fi
        ;;
    *)
        ;;
esac
])
