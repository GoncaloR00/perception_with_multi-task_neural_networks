; (function ($) {
	$.fn.SessionAlerter = function ( options ) {
		//parametros personalizáveis
		var settings = $.extend({
			maxSessionTime: 20,
			firstAlertTime: 5,
			secureArea: 'secure',
			homePage: 'Default.aspx',
			successHandler: '',
			pageToPing: ''
		}, options);

		var strings = {
			alert1: 'A sua sessão irá terminar dentro de {0} minutos. Quando a sessão terminar vai deixar de poder gravar a sua informação. Pode prolongar a sua sessão ou fechar esta janela para continuar.',
			alert2: 'A sua sessão expirou. Clique em <strong><em>fechar aviso</em></strong> para copiar quaisquer dados não gravados ou em <strong><em>início</em></strong> para voltar à página de entrada.',
			extendSession: 'prolongar sessão',
			sending: 'a enviar...',
			success: 'A sua sessão foi prolongada com sucesso.',
			error: 'Ocorreu um erro ao tentar contactar o servidor: ',
			home: 'início',
			reset: 'prolongar sessão',
			close: 'fechar aviso'
		};

		var css = {
			jqSessionAlerterOverlay: 'position:fixed;display:block;top:0;left:0;width:100%;height:100%;z-index:10000;background-color:#666;opacity:0.8',
			jqSessionAlerterBox: 'position:fixed;top:180px;left:{0}px;z-index:10001;width:{1}px;line-height:1.5em;color:#000;padding:12px {2}px;border:4px solid #333;background-color: #fff;text-align:left;',
			jqSessionAlerterButtonBar: 'text-align:right;padding-top:12px',
			jqSessionAlerterButton: 'padding:4px 6px;color:#333;border:1px solid #999;background-color:#eee;font-weight:bold;border-radius:2px;margin:4px'
		};

		//dimensões e posicionamento
		var width = 420;
		var padH = 16;
		var timer;

		var startSessionManager = function (context) {
			var url = document.URL.toLowerCase();
			
			// apenas mostra o alerta em páginas dentro do secure, exceto em páginas de popup (cujo nome começa com underscore)
			if (url.indexOf("/" + settings.secureArea) != -1 && url.indexOf("/" + settings.secureArea + "/_") == -1) {
				window.setTimeout(
					function () {
						// mostra mensagem
						ShowMessage(context, 1);
					}, (settings.maxSessionTime - settings.firstAlertTime) * 60 * 1000);
				
				timer = window.setTimeout(
					function () {
						// mostra mensagem
						ShowMessage(context, 2, settings, timer);
					}, settings.maxSessionTime * 60 * 1000);
			}
		};
		startSessionManager(this);

		// necessário para permitir o encadeamento de instruções
		return this;
		

		function ShowMessage(context, type) {
			var overlay = $('<div class="jqSessionAlerterOverlay" style="' + css.jqSessionAlerterOverlay + '"></div>');
			var left = Math.max(0, ((($(window).width() - (width + padH * 2)) / 2) + $(window).scrollLeft()));
			var style = css.jqSessionAlerterBox.replace('{0}', left).replace('{1}', width).replace('{2}', padH);

			var box = $('<div class="jqSessionAlerterBox" style="' + style + '"></div>');
			var buttons = $('<div class="jqSessionAlerterButtonBar" style="' + css.jqSessionAlerterButtonBar + '"></div>');

			if (type == 1) {
				// informa que a sessão está a terminar
				box.append('<span>' + strings.alert1.replace('{0}', settings.firstAlertTime) + '</span>');

				// desenha botão para prolongar a sessão (reset)
				var callTo = settings.pageToPing.length == 0 ? window.location : settings.pageToPing;
				buttons.append(DrawButton(strings.extendSession, 'reset').click(function (e) {
					e.preventDefault();
					$(this).off('click').html('<em>' + strings.sending + '</em>');
					$.ajax({ url: callTo })
						.done(function (data) {
							// mostra mensagem de sucesso
							WriteMessage(context, strings.success);

							// destroi mensagem e overlay após 2 segundos
							window.setTimeout(function () { DestroyWarning(context); }, 2 * 1000);

							// parar contador de fim de sessão
							window.clearTimeout(timer);

							// esconde botão de fechar janela
							$('.jqSessionAlerterButton.close').hide();

							// reinicializar contador
							startSessionManager(context);
							
							if(settings.successHandler != ''){
								if ($.isFunction(settings.successHandler)) {
									settings.successHandler();
								} else {
									$(document).trigger(settings.successHandler);
								}
							}
						})
						.fail(function (jqXHR, textStatus) {
							// mostra mensagem de erro
							WriteMessage(context, strings.error + textStatus);
						})
						.always(function () {
							// esconde botão de reset
							$('.jqSessionAlerterButton.reset').hide();
						})
				})).appendTo(box);

			} else {
				// esconde caixa com o primeiro aviso
				$('.jqSessionAlerterBox', context).remove();

				// mostra aviso que a sessão terminou
				box.append(strings.alert2);

				// desenha botão de voltar ao início
				buttons.append(DrawButton(strings.home, 'home').click(function (e) {
					e.preventDefault();
					window.location.href = settings.homePage;
				})).appendTo(box);
			}

			// adiciona botão para esconder mensagem
			buttons.append(DrawButton(strings.close, 'close').click(function (e) {
				e.preventDefault();
				DestroyWarning(context);
			})).appendTo(box);
			
			// adiciona os divs no container da mensagem
			DestroyWarning(context);
			context.append(overlay).append(box);
		}

		function WriteMessage(context, text){
			$('.jqSessionAlerterBox span', context).html(text);
		}

		function DestroyWarning(context) {
			$('.jqSessionAlerterBox', context).remove();
			$('.jqSessionAlerterOverlay', context).remove();
		}

		function DrawButton(text, ref) {
			return $('<a class="jqSessionAlerterButton ' + ref + '" style="' + css.jqSessionAlerterButton + '" href="#">' + text + '</a>');
		}

	};
}(jQuery));