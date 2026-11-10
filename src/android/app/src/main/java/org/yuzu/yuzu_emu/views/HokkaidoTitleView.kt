// SPDX-FileCopyrightText: Copyright 2026 Hokkaido contributors
// SPDX-License-Identifier: GPL-3.0-or-later

package org.yuzu.yuzu_emu.views

import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Shader
import android.graphics.Typeface
import android.util.AttributeSet
import androidx.core.content.ContextCompat
import com.google.android.material.textview.MaterialTextView
import org.yuzu.yuzu_emu.R

/** The main screen's "Hokkaido" title, filled with the logo's orange gradient. */
class HokkaidoTitleView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : MaterialTextView(context, attrs, defStyleAttr) {

    private val topColor = ContextCompat.getColor(context, R.color.hokkaido_title_top)
    private val bottomColor = ContextCompat.getColor(context, R.color.hokkaido_title_bottom)
    private var gradient: LinearGradient? = null
    private var gradientHeight = 0

    init {
        typeface = Typeface.create("sans-serif-black", Typeface.NORMAL)
        letterSpacing = 0.02f
        setShadowLayer(0f, 0f, 0f, 0)
    }

    override fun onDraw(canvas: Canvas) {
        if (gradient == null || gradientHeight != height) {
            gradientHeight = height
            gradient = LinearGradient(
                0f, paddingTop.toFloat(), 0f, (height - paddingBottom).toFloat(),
                topColor, bottomColor, Shader.TileMode.CLAMP
            )
        }
        paint.shader = gradient
        super.onDraw(canvas)
        paint.shader = null
    }
}
