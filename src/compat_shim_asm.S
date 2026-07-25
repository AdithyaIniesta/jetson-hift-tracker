/* Provide missing symbols at the assembly level to avoid C++ class redefinition.
 * These are no-op implementations of exception_ptr::_M_addref/_M_release. */

.text

.global _ZNSt15__exception_ptr13exception_ptr9_M_addrefEv
.type _ZNSt15__exception_ptr13exception_ptr9_M_addrefEv, %function
_ZNSt15__exception_ptr13exception_ptr9_M_addrefEv:
    ret

.global _ZNSt15__exception_ptr13exception_ptr10_M_releaseEv
.type _ZNSt15__exception_ptr13exception_ptr10_M_releaseEv, %function
_ZNSt15__exception_ptr13exception_ptr10_M_releaseEv:
    ret
