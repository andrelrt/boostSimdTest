#! /bin/sh

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`
cd $srcdir

autoreconf -vi || exit 1
cd $ORIGDIR || exit $?

$srcdir/configure "$@"
